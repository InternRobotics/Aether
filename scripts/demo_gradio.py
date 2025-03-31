import gc
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import gradio as gr
import imageio.v3 as iio
import numpy as np
import PIL
import rootutils
import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
from transformers import AutoTokenizer, T5EncoderModel


# import spaces

os.environ["GRADIO_TEMP_DIR"] = ".gradio_cache"
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from aether.pipelines.aetherv1_pipeline_cogvideox import (  # noqa: E402
    AetherV1PipelineCogVideoX,
    AetherV1PipelineOutput,
)
from aether.utils.postprocess_utils import (  # noqa: E402
    align_camera_extrinsics,
    apply_transformation,
    colorize_depth,
    compute_scale,
    get_intrinsics,
    interpolate_poses,
    postprocess_pointmap,
    project,
    raymap_to_poses,
)
from aether.utils.visualize_utils import predictions_to_glb  # noqa: E402


def seed_all(seed: int = 0) -> None:
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# # Global pipeline
cogvideox_pretrained_model_name_or_path: str = "THUDM/CogVideoX-5b-I2V"
aether_pretrained_model_name_or_path: str = "AetherWorldModel/AetherV1"
pipeline = AetherV1PipelineCogVideoX(
    tokenizer=AutoTokenizer.from_pretrained(
        cogvideox_pretrained_model_name_or_path,
        subfolder="tokenizer",
    ),
    text_encoder=T5EncoderModel.from_pretrained(
        cogvideox_pretrained_model_name_or_path, subfolder="text_encoder"
    ),
    vae=AutoencoderKLCogVideoX.from_pretrained(
        cogvideox_pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ),
    scheduler=CogVideoXDPMScheduler.from_pretrained(
        cogvideox_pretrained_model_name_or_path, subfolder="scheduler"
    ),
    transformer=CogVideoXTransformer3DModel.from_pretrained(
        aether_pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ),
)
pipeline.vae.enable_slicing()
pipeline.vae.enable_tiling()


def build_pipeline(device: torch.device) -> AetherV1PipelineCogVideoX:
    """Initialize the model pipeline."""
    pipeline.to(device)
    return pipeline


def get_window_starts(
    total_frames: int, sliding_window_size: int, temporal_stride: int
) -> List[int]:
    """Calculate window start indices."""
    starts = list(
        range(
            0,
            total_frames - sliding_window_size + 1,
            temporal_stride,
        )
    )
    if (
        total_frames > sliding_window_size
        and (total_frames - sliding_window_size) % temporal_stride != 0
    ):
        starts.append(total_frames - sliding_window_size)
    return starts


def blend_and_merge_window_results(
    window_results: List[AetherV1PipelineOutput], window_indices: List[int], args: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blend and merge window results."""
    merged_rgb = None
    merged_disparity = None
    merged_poses = None
    merged_focals = None
    align_pointmaps = args.get("align_pointmaps", True)
    smooth_camera = args.get("smooth_camera", True)
    smooth_method = args.get("smooth_method", "kalman") if smooth_camera else "none"

    if align_pointmaps:
        merged_pointmaps = None

    w1 = window_results[0].disparity

    for idx, (window_result, t_start) in enumerate(zip(window_results, window_indices)):
        t_end = t_start + window_result.rgb.shape[0]
        if idx == 0:
            merged_rgb = window_result.rgb
            merged_disparity = window_result.disparity
            pointmap_dict = postprocess_pointmap(
                window_result.disparity,
                window_result.raymap,
                vae_downsample_scale=8,
                ray_o_scale_inv=0.1,
                smooth_camera=smooth_camera,
                smooth_method=smooth_method if smooth_camera else "none",
            )
            merged_poses = pointmap_dict["camera_pose"]
            merged_focals = (
                pointmap_dict["intrinsics"][:, 0, 0]
                + pointmap_dict["intrinsics"][:, 1, 1]
            ) / 2
            if align_pointmaps:
                merged_pointmaps = pointmap_dict["pointmap"]
        else:
            overlap_t = window_indices[idx - 1] + window_result.rgb.shape[0] - t_start

            window_disparity = window_result.disparity

            # Align disparity
            disp_mask = window_disparity[:overlap_t].reshape(1, -1, w1.shape[-1]) > 0.1
            scale = compute_scale(
                window_disparity[:overlap_t].reshape(1, -1, w1.shape[-1]),
                merged_disparity[-overlap_t:].reshape(1, -1, w1.shape[-1]),
                disp_mask.reshape(1, -1, w1.shape[-1]),
            )
            window_disparity = scale * window_disparity

            # Blend disparity
            result_disparity = np.ones((t_end, *w1.shape[1:]))
            result_disparity[:t_start] = merged_disparity[:t_start]
            result_disparity[t_start + overlap_t :] = window_disparity[overlap_t:]
            weight = np.linspace(1, 0, overlap_t)[:, None, None]
            result_disparity[t_start : t_start + overlap_t] = merged_disparity[
                t_start : t_start + overlap_t
            ] * weight + window_disparity[:overlap_t] * (1 - weight)
            merged_disparity = result_disparity

            # Blend RGB
            result_rgb = np.ones((t_end, *w1.shape[1:], 3))
            result_rgb[:t_start] = merged_rgb[:t_start]
            result_rgb[t_start + overlap_t :] = window_result.rgb[overlap_t:]
            weight_rgb = np.linspace(1, 0, overlap_t)[:, None, None, None]
            result_rgb[t_start : t_start + overlap_t] = merged_rgb[
                t_start : t_start + overlap_t
            ] * weight_rgb + window_result.rgb[:overlap_t] * (1 - weight_rgb)
            merged_rgb = result_rgb

            # Align poses
            window_raymap = window_result.raymap
            window_poses, window_Fov_x, window_Fov_y = raymap_to_poses(
                window_raymap, ray_o_scale_inv=0.1
            )
            rel_r, rel_t, rel_s = align_camera_extrinsics(
                torch.from_numpy(window_poses[:overlap_t]),
                torch.from_numpy(merged_poses[-overlap_t:]),
            )
            aligned_window_poses = (
                apply_transformation(
                    torch.from_numpy(window_poses),
                    rel_r,
                    rel_t,
                    rel_s,
                    return_extri=True,
                )
                .cpu()
                .numpy()
            )

            result_poses = np.ones((t_end, 4, 4))
            result_poses[:t_start] = merged_poses[:t_start]
            result_poses[t_start + overlap_t :] = aligned_window_poses[overlap_t:]

            # Interpolate poses in overlap region
            weights = np.linspace(1, 0, overlap_t)
            for t in range(overlap_t):
                weight = weights[t]
                pose1 = merged_poses[t_start + t]
                pose2 = aligned_window_poses[t]
                result_poses[t_start + t] = interpolate_poses(pose1, pose2, weight)

            merged_poses = result_poses

            # Align intrinsics
            window_intrinsics, _ = get_intrinsics(
                batch_size=window_poses.shape[0],
                h=window_result.disparity.shape[1],
                w=window_result.disparity.shape[2],
                fovx=window_Fov_x,
                fovy=window_Fov_y,
            )
            window_focals = (
                window_intrinsics[:, 0, 0] + window_intrinsics[:, 1, 1]
            ) / 2
            scale = (merged_focals[-overlap_t:] / window_focals[:overlap_t]).mean()
            window_focals = scale * window_focals
            result_focals = np.ones((t_end,))
            result_focals[:t_start] = merged_focals[:t_start]
            result_focals[t_start + overlap_t :] = window_focals[overlap_t:]
            weight = np.linspace(1, 0, overlap_t)
            result_focals[t_start : t_start + overlap_t] = merged_focals[
                t_start : t_start + overlap_t
            ] * weight + window_focals[:overlap_t] * (1 - weight)
            merged_focals = result_focals

            if align_pointmaps:
                # Align pointmaps
                window_pointmaps = postprocess_pointmap(
                    result_disparity[t_start:],
                    window_raymap,
                    vae_downsample_scale=8,
                    camera_pose=aligned_window_poses,
                    focal=window_focals,
                    ray_o_scale_inv=0.1,
                    smooth_camera=smooth_camera,
                    smooth_method=smooth_method if smooth_camera else "none",
                )
                result_pointmaps = np.ones((t_end, *w1.shape[1:], 3))
                result_pointmaps[:t_start] = merged_pointmaps[:t_start]
                result_pointmaps[t_start + overlap_t :] = window_pointmaps["pointmap"][
                    overlap_t:
                ]
                weight = np.linspace(1, 0, overlap_t)[:, None, None, None]
                result_pointmaps[t_start : t_start + overlap_t] = merged_pointmaps[
                    t_start : t_start + overlap_t
                ] * weight + window_pointmaps["pointmap"][:overlap_t] * (1 - weight)
                merged_pointmaps = result_pointmaps

    # project to pointmaps
    height = args.get("height", 480)
    width = args.get("width", 720)

    intrinsics = [
        np.array([[f, 0, 0.5 * width], [0, f, 0.5 * height], [0, 0, 1]])
        for f in merged_focals
    ]
    if align_pointmaps:
        pointmaps = merged_pointmaps
    else:
        pointmaps = np.stack(
            [
                project(
                    1 / np.clip(merged_disparity[i], 1e-8, 1e8),
                    intrinsics[i],
                    merged_poses[i],
                )
                for i in range(merged_poses.shape[0])
            ]
        )

    return merged_rgb, merged_disparity, merged_poses, pointmaps


def process_video_to_frames(video_path: str, fps_sample: int = 12) -> List[str]:
    """Process video into frames and save them locally."""
    # Create a unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"temp_frames_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Read video
    video = iio.imread(video_path)

    # Calculate frame interval based on original video fps
    if isinstance(video, np.ndarray):
        # For captured videos
        total_frames = len(video)
        frame_interval = max(
            1, round(total_frames / (fps_sample * (total_frames / 30)))
        )
    else:
        # Default if can't determine
        frame_interval = 2

    frame_paths = []
    for i, frame in enumerate(video[::frame_interval]):
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        if isinstance(frame, np.ndarray):
            iio.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

    return frame_paths, output_dir


def save_output_files(
    rgb: np.ndarray,
    disparity: np.ndarray,
    poses: Optional[np.ndarray] = None,
    raymap: Optional[np.ndarray] = None,
    pointmap: Optional[np.ndarray] = None,
    task: str = "reconstruction",
    output_dir: str = "outputs",
    **kwargs,
) -> Dict[str, str]:
    """
    Save outputs and return paths to saved files.
    """
    os.makedirs(output_dir, exist_ok=True)

    if pointmap is None and raymap is not None:
        # # Generate pointmap from raymap and disparity
        # smooth_camera = kwargs.get("smooth_camera", True)
        # smooth_method = (
        #     kwargs.get("smooth_method", "kalman") if smooth_camera else "none"
        # )

        # pointmap_dict = postprocess_pointmap(
        #     disparity,
        #     raymap,
        #     vae_downsample_scale=8,
        #     ray_o_scale_inv=0.1,
        #     smooth_camera=smooth_camera,
        #     smooth_method=smooth_method,
        # )
        # pointmap = pointmap_dict["pointmap"]

        window_result = AetherV1PipelineOutput(
            rgb=rgb, disparity=disparity, raymap=raymap
        )
        window_results = [window_result]
        window_indices = [0]
        _, _, poses_from_blend, pointmap = blend_and_merge_window_results(
            window_results, window_indices, kwargs
        )

        # Use poses from blend_and_merge_window_results if poses is None
        if poses is None:
            poses = poses_from_blend

    if poses is None and raymap is not None:
        poses, _, _ = raymap_to_poses(raymap, ray_o_scale_inv=0.1)

    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{task}_{timestamp}"

    # Paths for saved files
    paths = {}

    # Save RGB video
    rgb_path = os.path.join(output_dir, f"{base_filename}_rgb.mp4")
    iio.imwrite(
        rgb_path,
        (np.clip(rgb, 0, 1) * 255).astype(np.uint8),
        fps=kwargs.get("fps", 12),
    )
    paths["rgb"] = rgb_path

    # Save depth/disparity video
    depth_path = os.path.join(output_dir, f"{base_filename}_disparity.mp4")
    iio.imwrite(
        depth_path,
        (colorize_depth(disparity) * 255).astype(np.uint8),
        fps=kwargs.get("fps", 12),
    )
    paths["disparity"] = depth_path

    # Save point cloud GLB files
    if pointmap is not None and poses is not None:
        pointcloud_save_frame_interval = kwargs.get(
            "pointcloud_save_frame_interval", 10
        )
        max_depth = kwargs.get("max_depth", 100.0)
        rtol = kwargs.get("rtol", 0.03)

        glb_paths = []
        # Determine which frames to save based on the interval
        frames_to_save = list(
            range(0, pointmap.shape[0], pointcloud_save_frame_interval)
        )

        # Always include the first and last frame
        if 0 not in frames_to_save:
            frames_to_save.insert(0, 0)
        if pointmap.shape[0] - 1 not in frames_to_save:
            frames_to_save.append(pointmap.shape[0] - 1)

        # Sort the frames to ensure they're in order
        frames_to_save = sorted(set(frames_to_save))

        for frame_idx in frames_to_save:
            if frame_idx >= pointmap.shape[0]:
                continue

            # fix the problem of point cloud being upside down and left-right reversed: flip Y axis and X axis
            flipped_pointmap = pointmap[frame_idx : frame_idx + 1].copy()
            flipped_pointmap[..., 1] = -flipped_pointmap[
                ..., 1
            ]  # flip Y axis (up and down)
            flipped_pointmap[..., 0] = -flipped_pointmap[
                ..., 0
            ]  # flip X axis (left and right)

            # flip camera poses
            flipped_poses = poses[frame_idx : frame_idx + 1].copy()
            # flip Y axis and X axis of camera orientation
            flipped_poses[..., 1, :3] = -flipped_poses[
                ..., 1, :3
            ]  # flip Y axis of camera orientation
            flipped_poses[..., 0, :3] = -flipped_poses[
                ..., 0, :3
            ]  # flip X axis of camera orientation
            flipped_poses[..., :3, 1] = -flipped_poses[
                ..., :3, 1
            ]  # flip Y axis of camera orientation
            flipped_poses[..., :3, 0] = -flipped_poses[
                ..., :3, 0
            ]  # flip X axis of camera orientation
            # flip Y axis and X axis of camera position
            flipped_poses[..., 1, 3] = -flipped_poses[..., 1, 3]  # flip Y axis position
            flipped_poses[..., 0, 3] = -flipped_poses[..., 0, 3]  # flip X axis position

            # use flipped point cloud and camera poses
            predictions = {
                "world_points": flipped_pointmap,
                "images": rgb[frame_idx : frame_idx + 1],
                "depths": 1 / np.clip(disparity[frame_idx : frame_idx + 1], 1e-8, 1e8),
                "camera_poses": flipped_poses,
            }

            glb_path = os.path.join(
                output_dir, f"{base_filename}_pointcloud_frame_{frame_idx}.glb"
            )

            scene_3d = predictions_to_glb(
                predictions,
                filter_by_frames="all",
                show_cam=True,
                max_depth=max_depth,
                rtol=rtol,
                frame_rel_idx=float(frame_idx) / pointmap.shape[0],
            )
            scene_3d.export(glb_path)
            glb_paths.append(glb_path)

        paths["pointcloud_glbs"] = glb_paths

    return paths


# @spaces.GPU(duration=300)
def process_reconstruction(
    video_file,
    height,
    width,
    num_frames,
    num_inference_steps,
    guidance_scale,
    sliding_window_stride,
    fps,
    smooth_camera,
    align_pointmaps,
    max_depth,
    rtol,
    pointcloud_save_frame_interval,
    seed,
    progress=gr.Progress(),
):
    """
    Process reconstruction task.
    """
    try:
        gc.collect()
        torch.cuda.empty_cache()

        seed_all(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Check your environment.")

        pipeline = build_pipeline(device)

        progress(0.1, "Loading video")
        # Check if video_file is a string or a file object
        if isinstance(video_file, str):
            video_path = video_file
        else:
            video_path = video_file.name

        video = iio.imread(video_path).astype(np.float32) / 255.0

        # Setup arguments
        args = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "sliding_window_stride": sliding_window_stride,
            "smooth_camera": smooth_camera,
            "smooth_method": "kalman" if smooth_camera else "none",
            "align_pointmaps": align_pointmaps,
            "max_depth": max_depth,
            "rtol": rtol,
            "pointcloud_save_frame_interval": pointcloud_save_frame_interval,
        }

        # Process in sliding windows
        window_results = []
        window_indices = get_window_starts(
            len(video), num_frames, sliding_window_stride
        )

        progress(0.2, f"Processing video in {len(window_indices)} windows")

        for i, start_idx in enumerate(window_indices):
            progress_val = 0.2 + (0.6 * (i / len(window_indices)))
            progress(progress_val, f"Processing window {i+1}/{len(window_indices)}")

            output = pipeline(
                task="reconstruction",
                image=None,
                goal=None,
                video=video[start_idx : start_idx + num_frames],
                raymap=None,
                height=height,
                width=width,
                num_frames=num_frames,
                fps=fps,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                use_dynamic_cfg=False,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            window_results.append(output)

        progress(0.8, "Merging results from all windows")
        # Merge window results
        (
            merged_rgb,
            merged_disparity,
            merged_poses,
            pointmaps,
        ) = blend_and_merge_window_results(window_results, window_indices, args)

        progress(0.9, "Saving output files")
        # Save output files
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_paths = save_output_files(
            rgb=merged_rgb,
            disparity=merged_disparity,
            poses=merged_poses,
            pointmap=pointmaps,
            task="reconstruction",
            output_dir=output_dir,
            fps=12,
            **args,
        )

        progress(1.0, "Done!")

        # Return paths for displaying
        return (
            output_paths["rgb"],
            output_paths["disparity"],
            output_paths.get("pointcloud_glbs", []),
        )

    except Exception:
        import traceback

        traceback.print_exc()
        return None, None, []


# @spaces.GPU(duration=300)
def process_prediction(
    image_file,
    height,
    width,
    num_frames,
    num_inference_steps,
    guidance_scale,
    use_dynamic_cfg,
    raymap_option,
    post_reconstruction,
    fps,
    smooth_camera,
    align_pointmaps,
    max_depth,
    rtol,
    pointcloud_save_frame_interval,
    seed,
    progress=gr.Progress(),
):
    """
    Process prediction task.
    """
    try:
        gc.collect()
        torch.cuda.empty_cache()

        # Set random seed
        seed_all(seed)

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Check your environment.")

        # Build the pipeline
        pipeline = build_pipeline(device)

        progress(0.1, "Loading image")
        # Check if image_file is a string or a file object
        if isinstance(image_file, str):
            image_path = image_file
        else:
            image_path = image_file.name

        image = PIL.Image.open(image_path)

        progress(0.2, "Running prediction")
        # Run prediction
        output = pipeline(
            task="prediction",
            image=image,
            video=None,
            goal=None,
            raymap=np.load(f"assets/example_raymaps/raymap_{raymap_option}.npy"),
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            generator=torch.Generator(device=device).manual_seed(seed),
            return_dict=True,
        )

        # Show RGB output immediately
        rgb_output = output.rgb

        # Setup arguments for saving
        args = {
            "height": height,
            "width": width,
            "smooth_camera": smooth_camera,
            "smooth_method": "kalman" if smooth_camera else "none",
            "align_pointmaps": align_pointmaps,
            "max_depth": max_depth,
            "rtol": rtol,
            "pointcloud_save_frame_interval": pointcloud_save_frame_interval,
        }

        if post_reconstruction:
            progress(0.5, "Running post-reconstruction for better quality")
            recon_output = pipeline(
                task="reconstruction",
                video=output.rgb,
                height=height,
                width=width,
                num_frames=num_frames,
                fps=fps,
                num_inference_steps=4,
                guidance_scale=1.0,
                use_dynamic_cfg=False,
                generator=torch.Generator(device=device).manual_seed(seed),
            )

            disparity = recon_output.disparity
            raymap = recon_output.raymap
        else:
            disparity = output.disparity
            raymap = output.raymap

        progress(0.8, "Saving output files")
        # Save output files
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_paths = save_output_files(
            rgb=rgb_output,
            disparity=disparity,
            raymap=raymap,
            task="prediction",
            output_dir=output_dir,
            fps=12,
            **args,
        )

        progress(1.0, "Done!")

        # Return paths for displaying
        return (
            output_paths["rgb"],
            output_paths["disparity"],
            output_paths.get("pointcloud_glbs", []),
        )

    except Exception:
        import traceback

        traceback.print_exc()
        return None, None, []


# @spaces.GPU(duration=300)
def process_planning(
    image_file,
    goal_file,
    height,
    width,
    num_frames,
    num_inference_steps,
    guidance_scale,
    use_dynamic_cfg,
    post_reconstruction,
    fps,
    smooth_camera,
    align_pointmaps,
    max_depth,
    rtol,
    pointcloud_save_frame_interval,
    seed,
    progress=gr.Progress(),
):
    """
    Process planning task.
    """
    try:
        gc.collect()
        torch.cuda.empty_cache()

        # Set random seed
        seed_all(seed)

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Check your environment.")

        # Build the pipeline
        pipeline = build_pipeline(device)

        progress(0.1, "Loading images")
        # Check if image_file and goal_file are strings or file objects
        if isinstance(image_file, str):
            image_path = image_file
        else:
            image_path = image_file.name

        if isinstance(goal_file, str):
            goal_path = goal_file
        else:
            goal_path = goal_file.name

        image = PIL.Image.open(image_path)
        goal = PIL.Image.open(goal_path)

        progress(0.2, "Running planning")
        # Run planning
        output = pipeline(
            task="planning",
            image=image,
            video=None,
            goal=goal,
            raymap=None,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            generator=torch.Generator(device=device).manual_seed(seed),
            return_dict=True,
        )

        # Show RGB output immediately
        rgb_output = output.rgb

        # Setup arguments for saving
        args = {
            "height": height,
            "width": width,
            "smooth_camera": smooth_camera,
            "smooth_method": "kalman" if smooth_camera else "none",
            "align_pointmaps": align_pointmaps,
            "max_depth": max_depth,
            "rtol": rtol,
            "pointcloud_save_frame_interval": pointcloud_save_frame_interval,
        }

        if post_reconstruction:
            progress(0.5, "Running post-reconstruction for better quality")
            recon_output = pipeline(
                task="reconstruction",
                video=output.rgb,
                height=height,
                width=width,
                num_frames=num_frames,
                fps=12,
                num_inference_steps=4,
                guidance_scale=1.0,
                use_dynamic_cfg=False,
                generator=torch.Generator(device=device).manual_seed(seed),
            )

            disparity = recon_output.disparity
            raymap = recon_output.raymap
        else:
            disparity = output.disparity
            raymap = output.raymap

        progress(0.8, "Saving output files")
        # Save output files
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_paths = save_output_files(
            rgb=rgb_output,
            disparity=disparity,
            raymap=raymap,
            task="planning",
            output_dir=output_dir,
            fps=fps,
            **args,
        )

        progress(1.0, "Done!")

        # Return paths for displaying
        return (
            output_paths["rgb"],
            output_paths["disparity"],
            output_paths.get("pointcloud_glbs", []),
        )

    except Exception:
        import traceback

        traceback.print_exc()
        return None, None, []


def update_task_ui(task):
    """Update UI elements based on selected task."""
    if task == "reconstruction":
        return (
            gr.update(visible=True),  # reconstruction_group
            gr.update(visible=False),  # prediction_group
            gr.update(visible=False),  # planning_group
            gr.update(visible=False),  # preview_row
            gr.update(value=4),  # num_inference_steps
            gr.update(visible=True),  # sliding_window_stride
            gr.update(visible=False),  # use_dynamic_cfg
            gr.update(visible=False),  # raymap_option
            gr.update(visible=False),  # post_reconstruction
            gr.update(value=1.0),  # guidance_scale
        )
    elif task == "prediction":
        return (
            gr.update(visible=False),  # reconstruction_group
            gr.update(visible=True),  # prediction_group
            gr.update(visible=False),  # planning_group
            gr.update(visible=True),  # preview_row
            gr.update(value=50),  # num_inference_steps
            gr.update(visible=False),  # sliding_window_stride
            gr.update(visible=True),  # use_dynamic_cfg
            gr.update(visible=True),  # raymap_option
            gr.update(visible=True),  # post_reconstruction
            gr.update(value=3.0),  # guidance_scale
        )
    elif task == "planning":
        return (
            gr.update(visible=False),  # reconstruction_group
            gr.update(visible=False),  # prediction_group
            gr.update(visible=True),  # planning_group
            gr.update(visible=True),  # preview_row
            gr.update(value=50),  # num_inference_steps
            gr.update(visible=False),  # sliding_window_stride
            gr.update(visible=True),  # use_dynamic_cfg
            gr.update(visible=False),  # raymap_option
            gr.update(visible=True),  # post_reconstruction
            gr.update(value=3.0),  # guidance_scale
        )


def update_image_preview(image_file):
    """Update the image preview."""
    if image_file is None:
        return None
    if isinstance(image_file, str):
        return image_file
    return image_file.name if hasattr(image_file, "name") else None


def update_goal_preview(goal_file):
    """Update the goal preview."""
    if goal_file is None:
        return None
    if isinstance(goal_file, str):
        return goal_file
    return goal_file.name if hasattr(goal_file, "name") else None


def get_download_link(selected_frame, all_paths):
    """Update the download button with the selected file path."""
    if not selected_frame or not all_paths:
        return gr.update(visible=False, value=None)

    frame_num = int(re.search(r"Frame (\d+)", selected_frame).group(1))

    for path in all_paths:
        if f"frame_{frame_num}" in path:
            # Make sure the file exists before setting it
            if os.path.exists(path):
                return gr.update(visible=True, value=path, interactive=True)

    return gr.update(visible=False, value=None)


# Theme setup
theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="cyan",
)

with gr.Blocks(
    theme=theme,
    css="""
    .output-column {
        min-height: 400px;
    }
    .warning {
        color: #856404 !important;
        font-weight: bold !important;
        padding: 10px !important;
        background-color: #fff3cd !important;
        border-left: 4px solid #ffc107 !important;
        border-radius: 4px !important;
        margin: 10px 0 !important;
    }
    .dark .warning {
        background-color: rgba(255, 193, 7, 0.1) !important;
        color: #fbd38d !important;
    }
    .highlight {
        background-color: rgba(0, 123, 255, 0.1);
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #007bff;
        margin: 10px 0;
    }
    .task-header {
        margin-top: 15px;
        margin-bottom: 20px;
        font-size: 1.4em;
        font-weight: bold;
        color: #007bff;
    }
    .flex-display {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .output-subtitle {
        font-size: 1.1em;
        margin-top: 5px;
        margin-bottom: 5px;
        color: #505050;
    }
    .input-section, .params-section, .advanced-section {
        border: 1px solid #ddd;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .logo-image {
        max-width: 300px;
        height: auto;
    }

    /* Optimize layout and spacing */
    .container {
        margin: 0 auto;
        padding: 0 15px;
        max-width: 1800px;
    }

    .header {
        text-align: center;
        margin-bottom: 20px;
        padding: 15px;
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        border-radius: 10px;
    }

    .dark .header {
        background: linear-gradient(to right, #2d3748, #1a202c);
    }

    .main-title {
        font-size: 2.2em;
        font-weight: bold;
        margin: 0 auto;
        color: #2c3e50;
        max-width: 800px;
    }

    .dark .main-title {
        color: #e2e8f0;
    }

    .links-bar {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin: 12px 0;
    }

    .link-button {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        background-color: #007bff;
        color: white !important;
        text-decoration: none;
        border-radius: 5px;
        transition: background-color 0.3s;
        font-size: 0.95em;
    }

    .link-button:hover {
        background-color: #0056b3;
        text-decoration: none;
    }

    .features-limitations-container {
        display: flex;
        gap: 15px;
        margin: 20px 0;
    }

    .capabilities-box, .limitations-box {
        flex: 1;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    .capabilities-box {
        background: #f0f9ff;
        border-left: 5px solid #3498db;
    }

    .dark .capabilities-box {
        background: #172a3a;
        border-left: 5px solid #3498db;
    }

    .limitations-box {
        background: #f8f9fa;
        border-left: 5px solid #ffc107;
    }

    .dark .limitations-box {
        background: #2d2a20;
        border-left: 5px solid #ffc107;
    }

    .capabilities-text, .limitations-text {
        color: #495057;
        line-height: 1.6;
    }

    .dark .capabilities-text, .dark .limitations-text {
        color: #cbd5e0;
    }

    .capabilities-text h3 {
        color: #2980b9;
        margin-top: 0;
        margin-bottom: 15px;
    }

    .dark .capabilities-text h3 {
        color: #63b3ed;
    }

    .limitations-text h3 {
        color: #d39e00;
        margin-top: 0;
        margin-bottom: 15px;
    }

    .dark .limitations-text h3 {
        color: #fbd38d;
    }

    .capabilities-text blockquote, .limitations-text blockquote {
        margin: 20px 0 0 0;
        padding: 10px 20px;
        font-style: italic;
    }

    .capabilities-text blockquote {
        border-left: 3px solid #3498db;
        background: rgba(52, 152, 219, 0.1);
    }

    .dark .capabilities-text blockquote {
        background: rgba(52, 152, 219, 0.2);
    }

    .limitations-text blockquote {
        border-left: 3px solid #ffc107;
        background: rgba(255, 193, 7, 0.1);
    }

    .dark .limitations-text blockquote {
        background: rgba(255, 193, 7, 0.2);
    }

    /* Optimize layout and spacing */
    .main-interface {
        display: flex;
        gap: 30px;
        margin-top: 20px;
    }

    .input-column, .output-column {
        flex: 1;
        min-width: 0;
        display: flex;
        flex-direction: column;
    }

    .output-panel {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        overflow-y: auto;
    }

    .dark .output-panel {
        border-color: #4a5568;
    }

    .run-button-container {
        display: flex;
        justify-content: center;
        margin: 15px 0;
    }

    .run-button {
        padding: 10px 30px;
        font-size: 1.1em;
        font-weight: bold;
        background: linear-gradient(to right, #3498db, #2980b9);
        border: none;
        border-radius: 5px;
        color: white;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .run-button:hover {
        background: linear-gradient(to right, #2980b9, #1a5276);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }

    .task-selector {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #e9ecef;
    }

    .dark .task-selector {
        background-color: #2d3748;
        border-color: #4a5568;
    }

    /* Compact parameter settings */
    .compact-params .row {
        margin-bottom: 8px;
    }

    .compact-params label {
        margin-bottom: 4px;
    }

    /* More obvious advanced options */
    .advanced-options-header {
        background-color: #e9ecef;
        padding: 10px 15px;
        border-radius: 6px;
        margin-top: 10px;
        font-weight: bold;
        color: #495057;
        border-left: 4px solid #6c757d;
        cursor: pointer;
        transition: all 0.2s;
    }

    .advanced-options-header:hover {
        background-color: #dee2e6;
    }

    .dark .advanced-options-header {
        background-color: #2d3748;
        color: #e2e8f0;
        border-left: 4px solid #a0aec0;
    }

    .dark .advanced-options-header:hover {
        background-color: #4a5568;
    }

    /* Vertical arrangement of output section */
    .output-section {
        margin-bottom: 30px;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 20px;
    }

    .output-section-title {
        font-weight: bold;
        color: #495057;
        margin-bottom: 15px;
        font-size: 1.2em;
    }

    .dark .output-section-title {
        color: #e2e8f0;
    }

    .pointcloud-controls {
        display: flex;
        gap: 10px;
        margin-bottom: 10px;
        align-items: center;
    }

    .note-box {
        background-color: #fff8e1 !important;
        border-left: 4px solid #ffc107 !important;
        padding: 12px !important;
        margin: 15px 0 !important;
        border-radius: 4px !important;
        color: #333 !important;
    }

    .dark .note-box {
        background-color: rgba(255, 193, 7, 0.1) !important;
        color: #e0e0e0 !important;
    }

    .note-box p, .note-box strong {
        color: inherit !important;
    }

    /* Ensure warning class styles are correctly applied */
    .warning {
        color: #856404 !important;
        font-weight: bold !important;
        padding: 10px !important;
        background-color: #fff3cd !important;
        border-left: 4px solid #ffc107 !important;
        border-radius: 4px !important;
        margin: 10px 0 !important;
    }

    .dark .warning {
        background-color: rgba(255, 193, 7, 0.1) !important;
        color: #fbd38d !important;
    }

    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px;
        margin: 15px 0;
        border-radius: 4px;
        color: #856404;
    }

    .dark .warning-box {
        background-color: rgba(255, 193, 7, 0.1);
        color: #fbd38d;
    }
""",
) as demo:
    with gr.Column(elem_classes=["container"]):
        with gr.Row(elem_classes=["header"]):
            with gr.Column():
                gr.Markdown(
                    """
                    # Aether: Geometric-Aware Unified World Modeling
                    """,
                    elem_classes=["main-title"],
                )

                gr.Markdown(
                    """
                    <div class="links-bar">
                        🌐<a href="https://aether-world.github.io/" class="link-button" target="_blank"> Project Page</a>
                        📄<a href="https://arxiv.org/abs/2503.18945" class="link-button" target="_blank"> Paper</a>
                        💻<a href="https://github.com/OpenRobotLab/Aether" class="link-button" target="_blank"> Code</a>
                        🤗<a href="https://huggingface.co/AetherWorldModel/AetherV1" class="link-button" target="_blank"> Model</a>
                    </div>
                    """,
                )

        with gr.Row(elem_classes=["features-limitations-container"]):
            with gr.Column(elem_classes=["capabilities-box"]):
                gr.Markdown(
                    """
                    ### 🚀 Key Capabilities

                    Aether addresses a fundamental challenge in AI: integrating geometric reconstruction with generative modeling for human-like spatial reasoning. Our framework unifies three core capabilities:

                    - 🌏 **4D Dynamic Reconstruction**: Reconstruct dynamic point clouds from videos by estimating depths and camera poses.

                    - 🎬 **Action-Conditioned Prediction**: Predict future frames based on initial observations, with optional camera trajectory actions.

                    - 🎯 **Goal-Conditioned Planning**: Generate planning paths from pairs of observation and goal images.

                    > *Trained entirely on synthetic data, Aether achieves strong zero-shot generalization to real-world scenarios.*
                    """,
                    elem_classes=["capabilities-text"],
                )

            with gr.Column(elem_classes=["limitations-box"]):
                gr.Markdown(
                    """
                    ### 📝 Current Limitations

                    Aether represents an initial step in our journey, trained entirely on synthetic data. While it demonstrates promising capabilities, it is important to be aware of its current limitations:

                    - 🔄 **Dynamic Scenarios**: Struggles with highly dynamic scenarios involving significant motion or dense crowds.

                    - 📸 **Camera Stability**: Camera pose estimation can be less stable in certain conditions.

                    - 📐 **Planning Range**: For visual planning tasks, we recommend keeping the observations and goals relatively close to ensure optimal performance.

                    > *We are actively working on the next generation of Aether and are committed to addressing these limitations in future releases.*
                    """,
                    elem_classes=["limitations-text"],
                )

        with gr.Row(elem_classes=["main-interface"]):
            with gr.Column(elem_classes=["input-column"]):
                with gr.Group(elem_classes=["task-selector"]):
                    task = gr.Radio(
                        ["reconstruction", "prediction", "planning"],
                        label="Select Task",
                        value="reconstruction",
                        info="Choose the task you want to perform",
                    )

                with gr.Group(elem_classes=["input-section"]):
                    gr.Markdown("## 📥 Input", elem_classes=["task-header"])

                    # Task-specific inputs
                    with gr.Group(visible=True) as reconstruction_group:
                        video_input = gr.Video(
                            label="Upload Input Video",
                            sources=["upload"],
                            interactive=True,
                            elem_id="video_input",
                        )
                        reconstruction_examples = gr.Examples(
                            examples=[
                                ["assets/example_videos/bridge.mp4"],
                                ["assets/example_videos/moviegen.mp4"],
                                ["assets/example_videos/nuscenes.mp4"],
                                ["assets/example_videos/veo2.mp4"],
                            ],
                            inputs=[video_input],
                            label="Reconstruction Examples",
                            examples_per_page=4,
                        )

                    with gr.Group(visible=False) as prediction_group:
                        image_input = gr.Image(
                            label="Upload Start Image",
                            type="filepath",
                            interactive=True,
                            elem_id="image_input",
                        )
                        prediction_examples = gr.Examples(
                            examples=[
                                ["assets/example_obs/car.png"],
                                ["assets/example_obs/cartoon.png"],
                                ["assets/example_obs/garden.jpg"],
                                ["assets/example_obs/room.jpg"],
                            ],
                            inputs=[image_input],
                            label="Prediction Examples",
                            examples_per_page=4,
                        )

                    with gr.Group(visible=False) as planning_group:
                        with gr.Row():
                            image_input_planning = gr.Image(
                                label="Upload Start Image",
                                type="filepath",
                                interactive=True,
                                elem_id="image_input_planning",
                            )
                            goal_input = gr.Image(
                                label="Upload Goal Image",
                                type="filepath",
                                interactive=True,
                                elem_id="goal_input",
                            )
                        planning_examples = gr.Examples(
                            examples=[
                                [
                                    "assets/example_obs_goal/01_obs.png",
                                    "assets/example_obs_goal/01_goal.png",
                                ],
                                [
                                    "assets/example_obs_goal/02_obs.png",
                                    "assets/example_obs_goal/02_goal.png",
                                ],
                                [
                                    "assets/example_obs_goal/03_obs.png",
                                    "assets/example_obs_goal/03_goal.png",
                                ],
                                [
                                    "assets/example_obs_goal/04_obs.png",
                                    "assets/example_obs_goal/04_goal.png",
                                ],
                            ],
                            inputs=[image_input_planning, goal_input],
                            label="Planning Examples",
                            examples_per_page=4,
                        )

                    with gr.Row(visible=False) as preview_row:
                        image_preview = gr.Image(
                            label="Start Image Preview",
                            elem_id="image_preview",
                            visible=False,
                        )
                        goal_preview = gr.Image(
                            label="Goal Image Preview",
                            elem_id="goal_preview",
                            visible=False,
                        )

                with gr.Group(elem_classes=["params-section", "compact-params"]):
                    gr.Markdown("## ⚙️ Parameters", elem_classes=["task-header"])

                    with gr.Row():
                        with gr.Column(scale=1):
                            height = gr.Dropdown(
                                choices=[480],
                                value=480,
                                label="Height",
                                info="Height of the output video",
                            )

                        with gr.Column(scale=1):
                            width = gr.Dropdown(
                                choices=[720],
                                value=720,
                                label="Width",
                                info="Width of the output video",
                            )

                    with gr.Row():
                        with gr.Column(scale=1):
                            num_frames = gr.Dropdown(
                                choices=[17, 25, 33, 41],
                                value=41,
                                label="Number of Frames",
                                info="Number of frames to predict",
                            )

                        with gr.Column(scale=1):
                            fps = gr.Dropdown(
                                choices=[8, 10, 12, 15, 24],
                                value=24,
                                label="FPS",
                                info="Frames per second",
                            )

                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            minimum=1,
                            maximum=60,
                            value=4,
                            step=1,
                            label="Inference Steps",
                            info="Number of inference step",
                        )

                    sliding_window_stride = gr.Slider(
                        minimum=1,
                        maximum=40,
                        value=24,
                        step=1,
                        label="Sliding Window Stride",
                        info="Sliding window stride (window size equals to num_frames). Only used for 'reconstruction' task",
                        visible=True,
                    )

                    use_dynamic_cfg = gr.Checkbox(
                        label="Use Dynamic CFG",
                        value=True,
                        info="Use dynamic CFG",
                        visible=False,
                    )

                    raymap_option = gr.Radio(
                        choices=["backward", "forward_right", "left_forward", "right"],
                        label="Camera Movement Direction",
                        value="forward_right",
                        info="Direction of camera action. We offer 4 pre-defined actions for you to choose from.",
                        visible=False,
                    )

                    post_reconstruction = gr.Checkbox(
                        label="Post-Reconstruction",
                        value=True,
                        info="Run reconstruction after prediction for better quality",
                        visible=False,
                    )

                    with gr.Accordion(
                        "Advanced Options",
                        open=False,
                        visible=True,
                        elem_classes=["advanced-options-header"],
                    ) as advanced_options:
                        with gr.Group(elem_classes=["advanced-section"]):
                            with gr.Row():
                                guidance_scale = gr.Slider(
                                    minimum=1.0,
                                    maximum=10.0,
                                    value=1.0,
                                    step=0.1,
                                    label="Guidance Scale",
                                    info="Guidance scale (only for prediction / planning)",
                                )

                            with gr.Row():
                                seed = gr.Number(
                                    value=42,
                                    label="Random Seed",
                                    info="Set a seed for reproducible results",
                                    precision=0,
                                    minimum=0,
                                    maximum=2147483647,
                                )

                            with gr.Row():
                                with gr.Column(scale=1):
                                    smooth_camera = gr.Checkbox(
                                        label="Smooth Camera",
                                        value=True,
                                        info="Apply smoothing to camera trajectory",
                                    )

                                with gr.Column(scale=1):
                                    align_pointmaps = gr.Checkbox(
                                        label="Align Point Maps",
                                        value=False,
                                        info="Align point maps across frames",
                                    )

                            with gr.Row():
                                with gr.Column(scale=1):
                                    max_depth = gr.Slider(
                                        minimum=10,
                                        maximum=200,
                                        value=60,
                                        step=10,
                                        label="Max Depth",
                                        info="Maximum depth for point cloud (higher = more distant points)",
                                    )

                                with gr.Column(scale=1):
                                    rtol = gr.Slider(
                                        minimum=0.01,
                                        maximum=2.0,
                                        value=0.2,
                                        step=0.01,
                                        label="Relative Tolerance",
                                        info="Used for depth edge detection. Lower = remove more edges",
                                    )

                            pointcloud_save_frame_interval = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=10,
                                step=1,
                                label="Point Cloud Frame Interval",
                                info="Save point cloud every N frames (higher = fewer files but less complete representation)",
                            )

                with gr.Group(elem_classes=["run-button-container"]):
                    run_button = gr.Button(
                        "Run Aether", variant="primary", elem_classes=["run-button"]
                    )

            with gr.Column(elem_classes=["output-column"]):
                with gr.Group(elem_classes=["output-panel"]):
                    gr.Markdown("## 📤 Output", elem_classes=["task-header"])

                    with gr.Group(elem_classes=["output-section"]):
                        gr.Markdown(
                            "### RGB Video", elem_classes=["output-section-title"]
                        )
                        rgb_output = gr.Video(
                            label="RGB Output", interactive=False, elem_id="rgb_output"
                        )

                    with gr.Group(elem_classes=["output-section"]):
                        gr.Markdown(
                            "### Depth Video", elem_classes=["output-section-title"]
                        )
                        depth_output = gr.Video(
                            label="Depth Output",
                            interactive=False,
                            elem_id="depth_output",
                        )

                    with gr.Group(elem_classes=["output-section"]):
                        gr.Markdown(
                            "### Point Clouds", elem_classes=["output-section-title"]
                        )
                        with gr.Row(elem_classes=["pointcloud-controls"]):
                            pointcloud_frames = gr.Dropdown(
                                label="Select Frame",
                                choices=[],
                                value=None,
                                interactive=True,
                                elem_id="pointcloud_frames",
                            )
                            pointcloud_download = gr.DownloadButton(
                                label="Download Point Cloud",
                                visible=False,
                                elem_id="pointcloud_download",
                            )

                        model_output = gr.Model3D(
                            label="Point Cloud Viewer",
                            interactive=True,
                            elem_id="model_output",
                        )

                        gr.Markdown(
                            """
                            > **Note:** 3D point clouds take a long time to visualize, and we show the keyframes only.
                            > You can control the keyframe interval by modifying the `pointcloud_save_frame_interval`.
                            """
                        )

                    with gr.Group(elem_classes=["output-section"]):
                        gr.Markdown(
                            "### About Results", elem_classes=["output-section-title"]
                        )
                        gr.Markdown(
                            """
                            #### Understanding the Outputs

                            - **RGB Video**: Shows the predicted or reconstructed RGB frames
                            - **Depth Video**: Visualizes the disparity maps in color (closer = red, further = blue)
                            - **Point Clouds**: Interactive 3D point cloud with camera positions shown as colored pyramids
                            """
                        )

    # Event handlers
    task.change(
        fn=update_task_ui,
        inputs=[task],
        outputs=[
            reconstruction_group,
            prediction_group,
            planning_group,
            preview_row,
            num_inference_steps,
            sliding_window_stride,
            use_dynamic_cfg,
            raymap_option,
            post_reconstruction,
            guidance_scale,
        ],
    )

    image_input.change(
        fn=update_image_preview, inputs=[image_input], outputs=[image_preview]
    ).then(fn=lambda: gr.update(visible=True), inputs=[], outputs=[preview_row])

    goal_input.change(
        fn=update_goal_preview, inputs=[goal_input], outputs=[goal_preview]
    ).then(fn=lambda: gr.update(visible=True), inputs=[], outputs=[preview_row])

    def update_pointcloud_frames(pointcloud_paths):
        """Update the pointcloud frames dropdown with available frames."""
        if not pointcloud_paths:
            return gr.update(choices=[], value=None), None, gr.update(visible=False)

        # Extract frame numbers from filenames
        frame_info = []
        for path in pointcloud_paths:
            filename = os.path.basename(path)
            match = re.search(r"frame_(\d+)", filename)
            if match:
                frame_num = int(match.group(1))
                frame_info.append((f"Frame {frame_num}", path))

        # Sort by frame number
        frame_info.sort(key=lambda x: int(re.search(r"Frame (\d+)", x[0]).group(1)))

        choices = [label for label, _ in frame_info]
        paths = [path for _, path in frame_info]

        if not choices:
            return gr.update(choices=[], value=None), None, gr.update(visible=False)

        # Make download button visible when we have point cloud files
        return (
            gr.update(choices=choices, value=choices[0]),
            paths[0],
            gr.update(visible=True),
        )

    def select_pointcloud_frame(frame_label, all_paths):
        """Select a specific pointcloud frame."""
        if not frame_label or not all_paths:
            return None

        frame_num = int(re.search(r"Frame (\d+)", frame_label).group(1))

        for path in all_paths:
            if f"frame_{frame_num}" in path:
                return path

        return None

    # Then in the run button click handler:
    def process_task(task_type, *args):
        """Process selected task with appropriate function."""
        if task_type == "reconstruction":
            rgb_path, depth_path, pointcloud_paths = process_reconstruction(*args)
            # Update the pointcloud frames dropdown
            frame_dropdown, initial_path, download_visible = update_pointcloud_frames(
                pointcloud_paths
            )
            return (
                rgb_path,
                depth_path,
                initial_path,
                frame_dropdown,
                pointcloud_paths,
                download_visible,
            )
        elif task_type == "prediction":
            rgb_path, depth_path, pointcloud_paths = process_prediction(*args)
            frame_dropdown, initial_path, download_visible = update_pointcloud_frames(
                pointcloud_paths
            )
            return (
                rgb_path,
                depth_path,
                initial_path,
                frame_dropdown,
                pointcloud_paths,
                download_visible,
            )
        elif task_type == "planning":
            rgb_path, depth_path, pointcloud_paths = process_planning(*args)
            frame_dropdown, initial_path, download_visible = update_pointcloud_frames(
                pointcloud_paths
            )
            return (
                rgb_path,
                depth_path,
                initial_path,
                frame_dropdown,
                pointcloud_paths,
                download_visible,
            )
        return (
            None,
            None,
            None,
            gr.update(choices=[], value=None),
            [],
            gr.update(visible=False),
        )

    # Store all pointcloud paths for later use
    all_pointcloud_paths = gr.State([])

    run_button.click(
        fn=lambda task_type,
        video_file,
        image_file,
        image_input_planning,
        goal_file,
        height,
        width,
        num_frames,
        num_inference_steps,
        guidance_scale,
        sliding_window_stride,
        use_dynamic_cfg,
        raymap_option,
        post_reconstruction,
        fps,
        smooth_camera,
        align_pointmaps,
        max_depth,
        rtol,
        pointcloud_save_frame_interval,
        seed: process_task(
            task_type,
            *(
                [
                    video_file,
                    height,
                    width,
                    num_frames,
                    num_inference_steps,
                    guidance_scale,
                    sliding_window_stride,
                    fps,
                    smooth_camera,
                    align_pointmaps,
                    max_depth,
                    rtol,
                    pointcloud_save_frame_interval,
                    seed,
                ]
                if task_type == "reconstruction"
                else [
                    image_file,
                    height,
                    width,
                    num_frames,
                    num_inference_steps,
                    guidance_scale,
                    use_dynamic_cfg,
                    raymap_option,
                    post_reconstruction,
                    fps,
                    smooth_camera,
                    align_pointmaps,
                    max_depth,
                    rtol,
                    pointcloud_save_frame_interval,
                    seed,
                ]
                if task_type == "prediction"
                else [
                    image_input_planning,
                    goal_file,
                    height,
                    width,
                    num_frames,
                    num_inference_steps,
                    guidance_scale,
                    use_dynamic_cfg,
                    post_reconstruction,
                    fps,
                    smooth_camera,
                    align_pointmaps,
                    max_depth,
                    rtol,
                    pointcloud_save_frame_interval,
                    seed,
                ]
            ),
        ),
        inputs=[
            task,
            video_input,
            image_input,
            image_input_planning,
            goal_input,
            height,
            width,
            num_frames,
            num_inference_steps,
            guidance_scale,
            sliding_window_stride,
            use_dynamic_cfg,
            raymap_option,
            post_reconstruction,
            fps,
            smooth_camera,
            align_pointmaps,
            max_depth,
            rtol,
            pointcloud_save_frame_interval,
            seed,
        ],
        outputs=[
            rgb_output,
            depth_output,
            model_output,
            pointcloud_frames,
            all_pointcloud_paths,
            pointcloud_download,
        ],
    )

    pointcloud_frames.change(
        fn=select_pointcloud_frame,
        inputs=[pointcloud_frames, all_pointcloud_paths],
        outputs=[model_output],
    ).then(
        fn=get_download_link,
        inputs=[pointcloud_frames, all_pointcloud_paths],
        outputs=[pointcloud_download],
    )

    # Load the model at startup
    demo.load(lambda: build_pipeline(torch.device("cpu")), inputs=None, outputs=None)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    demo.queue(max_size=20).launch(show_error=True, share=False, server_port=7860)
