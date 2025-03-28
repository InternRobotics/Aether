import inspect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.image_processor import PipelineImageInput
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import AutoTokenizer, T5EncoderModel

from aether.utils.preprocess_utils import imcrop_center


def get_3d_rotary_pos_embed(
    embed_dim,
    crops_coords,
    grid_size,
    temporal_size,
    theta: int = 10000,
    use_real: bool = True,
    grid_type: str = "linspace",
    max_size: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    fps_factor: Optional[float] = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.
    grid_type (`str`):
        Whether to use "linspace" or "slice" to compute grids.
    fps_factor (`float`):
        The relative fps factor of the video, computed by base_fps / fps. Useful for variable fps training.

    Returns:
        `torch.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    if use_real is not True:
        raise ValueError(
            " `use_real = False` is not currently supported for get_3d_rotary_pos_embed"
        )

    if grid_type == "linspace":
        start, stop = crops_coords
        grid_size_h, grid_size_w = grid_size
        grid_h = torch.linspace(
            start[0],
            stop[0] * (grid_size_h - 1) / grid_size_h,
            grid_size_h,
            device=device,
            dtype=torch.float32,
        )
        grid_w = torch.linspace(
            start[1],
            stop[1] * (grid_size_w - 1) / grid_size_w,
            grid_size_w,
            device=device,
            dtype=torch.float32,
        )
        grid_t = (
            torch.linspace(
                0,
                temporal_size * (temporal_size - 1) / temporal_size,
                temporal_size,
                device=device,
                dtype=torch.float32,
            )
            * fps_factor
        )
    elif grid_type == "slice":
        max_h, max_w = max_size
        grid_size_h, grid_size_w = grid_size
        grid_h = torch.arange(max_h, device=device, dtype=torch.float32)
        grid_w = torch.arange(max_w, device=device, dtype=torch.float32)
        grid_t = (
            torch.arange(temporal_size, device=device, dtype=torch.float32) * fps_factor
        )
    else:
        raise ValueError("Invalid value passed for `grid_type`.")

    # Compute dimensions for each axis
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # Temporal frequencies
    freqs_t = get_1d_rotary_pos_embed(dim_t, grid_t, theta=theta, use_real=True)
    # Spatial frequencies for height and width
    freqs_h = get_1d_rotary_pos_embed(dim_h, grid_h, theta=theta, use_real=True)
    freqs_w = get_1d_rotary_pos_embed(dim_w, grid_w, theta=theta, use_real=True)

    # BroadCast and concatenate temporal and spaial frequencie (height and width) into a 3d tensor
    def combine_time_height_width(freqs_t, freqs_h, freqs_w):
        freqs_t = freqs_t[:, None, None, :].expand(
            -1, grid_size_h, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_w, dim_t
        freqs_h = freqs_h[None, :, None, :].expand(
            temporal_size, -1, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_h
        freqs_w = freqs_w[None, None, :, :].expand(
            temporal_size, grid_size_h, -1, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_w

        freqs = torch.cat(
            [freqs_t, freqs_h, freqs_w], dim=-1
        )  # temporal_size, grid_size_h, grid_size_w, (dim_t + dim_h + dim_w)
        freqs = freqs.view(
            temporal_size * grid_size_h * grid_size_w, -1
        )  # (temporal_size * grid_size_h * grid_size_w), (dim_t + dim_h + dim_w)
        return freqs

    t_cos, t_sin = freqs_t  # both t_cos and t_sin has shape: temporal_size, dim_t
    h_cos, h_sin = freqs_h  # both h_cos and h_sin has shape: grid_size_h, dim_h
    w_cos, w_sin = freqs_w  # both w_cos and w_sin has shape: grid_size_w, dim_w

    if grid_type == "slice":
        t_cos, t_sin = t_cos[:temporal_size], t_sin[:temporal_size]
        h_cos, h_sin = h_cos[:grid_size_h], h_sin[:grid_size_h]
        w_cos, w_sin = w_cos[:grid_size_w], w_sin[:grid_size_w]

    cos = combine_time_height_width(t_cos, h_cos, w_cos)
    sin = combine_time_height_width(t_sin, h_sin, w_sin)
    return cos, sin


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


@dataclass
class AetherV1PipelineOutput(BaseOutput):
    rgb: np.ndarray
    disparity: np.ndarray
    raymap: np.ndarray


class AetherV1PipelineCogVideoX(CogVideoXImageToVideoPipeline):
    _supported_tasks = ["reconstruction", "prediction", "planning"]
    _default_num_inference_steps = {
        "reconstruction": 4,
        "prediction": 50,
        "planning": 50,
    }
    _default_guidance_scale = {
        "reconstruction": 1.0,
        "prediction": 3.0,
        "planning": 3.0,
    }
    _default_use_dynamic_cfg = {
        "reconstruction": False,
        "prediction": True,
        "planning": True,
    }
    _base_fps = 12

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        scheduler: CogVideoXDPMScheduler,
        transformer: CogVideoXTransformer3DModel,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
        )

        self.empty_prompt_embeds, _ = self.encode_prompt(
            prompt="",
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            prompt_embeds=None,
        )
        self.empty_prompt_embeds = self.empty_prompt_embeds.to(dtype=torch.bfloat16)

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
        fps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (
            self.vae_scale_factor_spatial * self.transformer.config.patch_size
        )
        grid_width = width // (
            self.vae_scale_factor_spatial * self.transformer.config.patch_size
        )

        p = self.transformer.config.patch_size
        p_t = self.transformer.config.patch_size_t

        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p

        if p_t is None:
            # CogVideoX 1.0
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
                fps_factor=self._base_fps / fps,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
                fps_factor=self._base_fps / fps,
            )

        return freqs_cos, freqs_sin

    def check_inputs(
        self,
        task,
        image,
        video,
        goal,
        raymap,
        height,
        width,
        num_frames,
        fps,
    ):
        if task not in self._supported_tasks:
            raise ValueError(f"`task` has to be one of {self._supported_tasks}.")

        if image is None and video is None:
            raise ValueError("`image` or `video` has to be provided.")

        if image is not None and video is not None:
            raise ValueError("`image` and `video` cannot both be provided.")

        if image is not None:
            if task == "reconstruction":
                raise ValueError("`image` is not supported for `reconstruction` task.")
            if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, np.ndarray)
                and not isinstance(image, PIL.Image.Image)
            ):
                raise ValueError(
                    "`image` has to be of type `torch.Tensor` or `np.ndarray` or `PIL.Image.Image` but is"
                    f" {type(image)}"
                )

        if goal is not None:
            if task != "planning":
                raise ValueError("`goal` is only supported for `planning` task.")

            if (
                not isinstance(goal, torch.Tensor)
                and not isinstance(goal, np.ndarray)
                and not isinstance(goal, PIL.Image.Image)
            ):
                raise ValueError(
                    "`goal` has to be of type `torch.Tensor` or `np.ndarray` or `PIL.Image.Image` but is"
                    f" {type(goal)}"
                )

        if video is not None:
            if task != "reconstruction":
                raise ValueError("`video` is only supported for `reconstruction` task.")

            if (
                not isinstance(video, torch.Tensor)
                and not isinstance(video, np.ndarray)
                and not (
                    isinstance(video, list)
                    and all(isinstance(v, PIL.Image.Image) for v in video)
                )
            ):
                raise ValueError(
                    "`video` has to be of type `torch.Tensor` or `np.ndarray` or `List[PIL.Image.Image]` but is"
                    f" {type(video)}"
                )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if num_frames is None:
            raise ValueError("`num_frames` is required.")

        if num_frames not in [17, 25, 33, 41]:
            raise ValueError("`num_frames` has to be one of [17, 25, 33, 41].")

        if fps not in [8, 10, 12, 15, 24]:
            raise ValueError("`fps` has to be one of [8, 10, 12, 15, 24].")

        if (
            raymap is not None
            and not isinstance(raymap, torch.Tensor)
            and not isinstance(raymap, np.ndarray)
        ):
            raise ValueError(
                "`raymap` has to be of type `torch.Tensor` or `np.ndarray`."
            )

        if raymap is not None:
            if raymap.shape[-4:] != (
                num_frames,
                6,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            ):
                raise ValueError(
                    f"`raymap` shape is not correct. "
                    f"Expected {num_frames, 6, height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial}, "
                    f"got {raymap.shape}."
                )

    def _preprocess_image(self, image, height, width):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if image.ndim == 3:
            image = [image]
        image = imcrop_center(image, height, width)
        image = self.video_processor.preprocess(image, height, width)
        return image

    def preprocess_inputs(
        self,
        image,
        goal,
        video,
        raymap,
        height,
        width,
        num_frames,
    ):
        if image is not None:
            if isinstance(image, PIL.Image.Image):
                image = self.video_processor.preprocess(
                    image, height, width, resize_mode="crop"
                ).to(device=self._execution_device, dtype=torch.bfloat16)
            else:
                image = self._preprocess_image(image, height, width).to(
                    device=self._execution_device, dtype=torch.bfloat16
                )
        if goal is not None:
            if isinstance(goal, PIL.Image.Image):
                goal = self.video_processor.preprocess(
                    goal, height, width, resize_mode="crop"
                ).to(device=self._execution_device, dtype=torch.bfloat16)
            else:
                goal = self._preprocess_image(goal, height, width).to(
                    device=self._execution_device, dtype=torch.bfloat16
                )
        if video is not None:
            if isinstance(video, list) and all(
                isinstance(v, PIL.Image.Image) for v in video
            ):
                video = self.video_processor.preprocess(
                    video, height, width, resize_mode="crop"
                ).to(device=self._execution_device, dtype=torch.bfloat16)
            else:
                video = self._preprocess_image(video, height, width).to(
                    device=self._execution_device, dtype=torch.bfloat16
                )
        # TODO: check raymap shape
        if raymap is not None:
            if isinstance(raymap, np.ndarray):
                raymap = torch.from_numpy(raymap).to(
                    self._execution_device, dtype=torch.bfloat16
                )
            if raymap.ndim == 4:
                raymap = raymap.unsqueeze(0).to(
                    self._execution_device, dtype=torch.bfloat16
                )

        return image, goal, video, raymap

    @torch.no_grad()
    def prepare_latents(
        self,
        image: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        raymap: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_frames: int = 13,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            56,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # For CogVideoX1.5, the latent should add 1 for padding (Not use)
        if self.transformer.config.patch_size_t is not None:
            shape = (
                shape[:1]
                + (shape[1] + shape[1] % self.transformer.config.patch_size_t,)
                + shape[2:]
            )

        if image is not None:
            image = image.unsqueeze(2)
            if isinstance(generator, list):
                image_latents = [
                    retrieve_latents(
                        self.vae.encode(image[i].unsqueeze(0)), generator[i]
                    )
                    for i in range(batch_size)
                ]
            else:
                image_latents = [
                    retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator)
                    for img in image
                ]

            image_latents = (
                torch.cat(image_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)
            )  # [B, F, C, H, W]

            if not self.vae.config.invert_scale_latents:
                image_latents = self.vae_scaling_factor_image * image_latents
            else:
                # This is awkward but required because the CogVideoX team forgot to multiply the
                # scaling factor during training :)
                image_latents = 1 / self.vae_scaling_factor_image * image_latents

        if goal is not None:
            goal = goal.unsqueeze(2)
            if isinstance(generator, list):
                goal_latents = [
                    retrieve_latents(
                        self.vae.encode(goal[i].unsqueeze(0)), generator[i]
                    )
                    for i in range(batch_size)
                ]
            else:
                goal_latents = [
                    retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator)
                    for img in goal
                ]

            goal_latents = (
                torch.cat(goal_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)
            )  # [B, F, C, H, W]

            if not self.vae.config.invert_scale_latents:
                goal_latents = self.vae_scaling_factor_image * goal_latents
            else:
                # This is awkward but required because the CogVideoX team forgot to multiply the
                # scaling factor during training :)
                goal_latents = 1 / self.vae_scaling_factor_image * goal_latents

        if video is not None:
            if video.ndim == 4:
                video = video.unsqueeze(0)

            video = video.permute(0, 2, 1, 3, 4)
            if isinstance(generator, list):
                video_latents = [
                    retrieve_latents(
                        self.vae.encode(video[i].unsqueeze(0)), generator[i]
                    )
                    for i in range(batch_size)
                ]
            else:
                video_latents = [
                    retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator)
                    for img in video
                ]

            video_latents = (
                torch.cat(video_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)
            )  # [B, F, C, H, W]

            if not self.vae.config.invert_scale_latents:
                video_latents = self.vae_scaling_factor_image * video_latents
            else:
                # This is awkward but required because the CogVideoX team forgot to multiply the
                # scaling factor during training :)
                video_latents = 1 / self.vae_scaling_factor_image * video_latents

        if image is not None and goal is None:
            padding_shape = (
                batch_size,
                num_frames - image_latents.shape[1],
                *image_latents.shape[2:],
            )
            padding = torch.zeros(padding_shape, device=device, dtype=dtype)
            condition_latents = torch.cat([image_latents, padding], dim=1)
        elif goal is not None:
            padding_shape = (
                batch_size,
                num_frames - goal_latents.shape[1] - image_latents.shape[1],
                *image_latents.shape[2:],
            )
            padding = torch.zeros(padding_shape, device=device, dtype=dtype)
            condition_latents = torch.cat([image_latents, padding, goal_latents], dim=1)
        elif video is not None:
            condition_latents = video_latents

        if raymap is not None:
            if raymap.shape[1] % self.vae_scale_factor_temporal != 0:
                # repeat
                raymap = torch.cat(
                    [
                        raymap[
                            :,
                            : self.vae_scale_factor_temporal
                            - raymap.shape[1] % self.vae_scale_factor_temporal,
                        ],
                        raymap,
                    ],
                    dim=1,
                )
            camera_conditions = rearrange(
                raymap,
                "b (n t) c h w -> b t (n c) h w",
                n=self.vae_scale_factor_temporal,
            )
        else:
            camera_conditions = torch.zeros(
                batch_size,
                num_frames,
                24,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
                device=device,
                dtype=dtype,
            )

        condition_latents = torch.cat([condition_latents, camera_conditions], dim=2)
        latents = randn_tensor(shape, device=device, generator=generator, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        return latents, condition_latents

    @torch.no_grad()
    def __call__(
        self,
        task: Optional[str] = None,
        image: Optional[PipelineImageInput] = None,
        video: Optional[PipelineImageInput] = None,
        goal: Optional[PipelineImageInput] = None,
        raymap: Optional[Union[torch.Tensor, np.ndarray]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        guidance_scale: Optional[float] = None,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict] = None,
        fps: Optional[int] = None,
    ) -> Union[AetherV1PipelineOutput, Tuple]:
        if task is None:
            if video is not None:
                task = "reconstruction"
            elif goal is not None:
                task = "planning"
            else:
                task = "prediction"

        height = (
            height
            or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        )
        width = (
            width
            or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        )
        num_frames = num_frames or self.transformer.config.sample_frames
        fps = fps or self._base_fps

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            task=task,
            image=image,
            video=video,
            goal=goal,
            raymap=raymap,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )

        # 2. Preprocess inputs
        image, goal, video, raymap = self.preprocess_inputs(
            image=image,
            goal=goal,
            video=video,
            raymap=raymap,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        batch_size = 1

        device = self._execution_device

        # 3. Encode input prompt
        prompt_embeds = self.empty_prompt_embeds.to(device)

        num_inference_steps = (
            num_inference_steps or self._default_num_inference_steps[task]
        )
        guidance_scale = guidance_scale or self._default_guidance_scale[task]
        use_dynamic_cfg = use_dynamic_cfg or self._default_use_dynamic_cfg[task]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latents, condition_latents = self.prepare_latents(
            image,
            goal,
            video,
            raymap,
            batch_size * num_videos_per_prompt,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(
                height, width, latents.size(1), device, fps=fps
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Create ofs embeds if required
        ofs_emb = (
            None
            if self.transformer.config.ofs_embed_dim is None
            else latents.new_full((1,), fill_value=2.0)
        )

        # 8. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if do_classifier_free_guidance:
                    if task == "planning":
                        assert goal is not None
                        uncond = condition_latents.clone()
                        uncond[:, :, : self.vae.config.latent_channels] = 0
                        latent_condition = torch.cat([uncond, condition_latents])
                    elif task == "prediction":
                        uncond = condition_latents.clone()
                        uncond[:, :1, : self.vae.config.latent_channels] = 0
                        latent_condition = torch.cat([uncond, condition_latents])
                    else:
                        raise ValueError(
                            f"Task {task} not supported for classifier-free guidance."
                        )

                else:
                    latent_condition = condition_latents

                latent_model_input = torch.cat(
                    [latent_model_input, latent_condition], dim=2
                )

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds.repeat(
                        latent_model_input.shape[0], 1, 1
                    ),
                    timestep=timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (
                            1
                            - math.cos(
                                math.pi
                                * (
                                    (num_inference_steps - t.item())
                                    / num_inference_steps
                                )
                                ** 5.0
                            )
                        )
                        / 2
                    )

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

        rgb_latents = latents[:, :, : self.vae.config.latent_channels]
        disparity_latents = latents[
            :, :, self.vae.config.latent_channels : self.vae.config.latent_channels * 2
        ]
        camera_latents = latents[:, :, self.vae.config.latent_channels * 2 :]

        rgb_video = self.decode_latents(rgb_latents)
        rgb_video = self.video_processor.postprocess_video(
            video=rgb_video, output_type="np"
        )

        disparity_video = self.decode_latents(disparity_latents)
        disparity_video = disparity_video.mean(dim=1, keepdim=False)
        disparity_video = disparity_video * 0.5 + 0.5
        disparity_video = torch.square(disparity_video)
        disparity_video = disparity_video.float().cpu().numpy()

        raymap = (
            rearrange(camera_latents, "b t (n c) h w -> b (n t) c h w", n=4)[
                :, -rgb_video.shape[1] :, :, :
            ]
            .float()
            .cpu()
            .numpy()
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (
                rgb_video,
                disparity_video,
                raymap,
            )

        return AetherV1PipelineOutput(
            rgb=rgb_video.squeeze(0),
            disparity=disparity_video.squeeze(0),
            raymap=raymap.squeeze(0),
        )
