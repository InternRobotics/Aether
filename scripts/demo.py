import argparse
import os
import random
from typing import List, Optional, Tuple

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int = 0) -> None:
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AetherV1-CogvideoX Inference Demo")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["reconstruction", "prediction", "planning"],
        help="Task to perform: 'reconstruction', 'prediction' or 'planning'.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file. Only used for 'reconstruction' task.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file. Only used for 'prediction' and 'planning' tasks.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Path to a goal image file. Only used for 'planning' task.",
    )
    parser.add_argument(
        "--raymap_action",
        type=str,
        default=None,
        help="Path to a raymap action file. Should be a numpy array of shape (num_frame, 6, latent_height, latent_width).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Path to save the outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        choices=[8, 10, 12, 15, 24],
        help="Frames per second. Options: 8, 10, 12, 15, 24.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of inference steps. If not specified, will use the default number of steps for the task.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale. If not specified, will use the default guidance scale for the task.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=True,
        help="Use dynamic cfg.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of the output video.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Width of the output video.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=41,
        help="Number of frames to predict.",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=100.0,
        help="Maximum depth of the scene in meters.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.2,
        help="Relative tolerance for depth edge detection.",
    )
    parser.add_argument(
        "--cogvideox_pretrained_model_name_or_path",
        type=str,
        default="THUDM/CogVideoX-5b-I2V",
        help="Name or path of the CogVideoX model to use.",
    )
    parser.add_argument(
        "--aether_pretrained_model_name_or_path",
        type=str,
        default="AetherWorldModel/AetherV1",
        help="Name or path of the Aether model to use.",
    )
    parser.add_argument(
        "--smooth_camera",
        action="store_true",
        default=True,
        help="Smooth the camera trajectory.",
    )
    parser.add_argument(
        "--smooth_method",
        type=str,
        default="kalman",
        choices=["kalman", "simple"],
        help="Smooth method.",
    )
    parser.add_argument(
        "--sliding_window_stride",
        type=int,
        default=24,
        help="Sliding window stride (window size equals to num_frames). Only used for 'reconstruction' task.",
    )
    parser.add_argument(
        "--post_reconstruction",
        action="store_true",
        default=True,
        help="Run reconstruction after prediction for better quality. Only used for 'prediction' and 'planning' tasks.",
    )
    parser.add_argument(
        "--pointcloud_save_frame_interval",
        type=int,
        default=10,
        help="Pointcloud save frame interval.",
    )
    parser.add_argument(
        "--align_pointmaps",
        action="store_true",
        default=False,
        help="Align pointmaps.",
    )
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> AetherV1PipelineCogVideoX:
    pipeline = AetherV1PipelineCogVideoX(
        tokenizer=AutoTokenizer.from_pretrained(
            args.cogvideox_pretrained_model_name_or_path,
            subfolder="tokenizer",
        ),
        text_encoder=T5EncoderModel.from_pretrained(
            args.cogvideox_pretrained_model_name_or_path, subfolder="text_encoder"
        ),
        vae=AutoencoderKLCogVideoX.from_pretrained(
            args.cogvideox_pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ),
        scheduler=CogVideoXDPMScheduler.from_pretrained(
            args.cogvideox_pretrained_model_name_or_path, subfolder="scheduler"
        ),
        transformer=CogVideoXTransformer3DModel.from_pretrained(
            args.aether_pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ),
    )
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
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
    window_results: List[AetherV1PipelineOutput],
    window_indices: List[int],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blend and merge window results."""
    merged_rgb = None
    merged_disparity = None
    merged_poses = None
    merged_focals = None
    if args.align_pointmaps:
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
                smooth_camera=args.smooth_camera,
                smooth_method=args.smooth_method if args.smooth_camera else "none",
            )
            merged_poses = pointmap_dict["camera_pose"]
            merged_focals = (
                pointmap_dict["intrinsics"][:, 0, 0]
                + pointmap_dict["intrinsics"][:, 1, 1]
            ) / 2
            if args.align_pointmaps:
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

            if args.align_pointmaps:
                # Align pointmaps
                window_pointmaps = postprocess_pointmap(
                    result_disparity[t_start:],
                    window_raymap,
                    vae_downsample_scale=8,
                    camera_pose=aligned_window_poses,
                    focal=window_focals,
                    ray_o_scale_inv=0.1,
                    smooth_camera=args.smooth_camera,
                    smooth_method=args.smooth_method if args.smooth_camera else "none",
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
    intrinsics = [
        np.array([[f, 0, 0.5 * args.width], [0, f, 0.5 * args.height], [0, 0, 1]])
        for f in merged_focals
    ]
    if args.align_pointmaps:
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


def save_output(
    rgb: np.ndarray,
    disparity: np.ndarray,
    poses: Optional[np.ndarray] = None,
    raymap: Optional[np.ndarray] = None,
    pointmap: Optional[np.ndarray] = None,
    args: argparse.Namespace = None,
) -> None:
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if pointmap is None:
        assert raymap is not None, "Raymap is required for saving pointmap."
        window_result = AetherV1PipelineOutput(
            rgb=rgb, disparity=disparity, raymap=raymap
        )
        window_results = [window_result]
        window_indices = [0]
        _, _, poses_from_blend, pointmap = blend_and_merge_window_results(
            window_results, window_indices, args
        )

        # Use poses from blend_and_merge_window_results if poses is None
        if poses is None:
            poses = poses_from_blend

    if poses is None:
        assert raymap is not None, "Raymap is required for saving poses."
        poses, _, _ = raymap_to_poses(raymap, ray_o_scale_inv=0.1)

    # Fix the problem of point cloud being upside down and left-right reversed
    # Flip Y axis and X axis for both pointmap and camera poses
    flipped_pointmap = pointmap.copy()
    flipped_pointmap[..., 1] = -flipped_pointmap[..., 1]  # flip Y axis (up and down)
    flipped_pointmap[..., 0] = -flipped_pointmap[..., 0]  # flip X axis (left and right)

    # Flip camera poses
    flipped_poses = poses.copy()
    # Flip Y axis and X axis of camera orientation
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
    # Flip Y axis and X axis of camera position
    flipped_poses[..., 1, 3] = -flipped_poses[..., 1, 3]  # flip Y axis position
    flipped_poses[..., 0, 3] = -flipped_poses[..., 0, 3]  # flip X axis position

    # Use the flipped versions for output
    pointmap = flipped_pointmap
    poses = flipped_poses

    if args.task == "reconstruction":
        filename = f"reconstruction_{args.video.split('/')[-1].split('.')[0]}"
    elif args.task == "prediction":
        filename = f"prediction_{args.image.split('/')[-1].split('.')[0]}"
    elif args.task == "planning":
        filename = f"planning_{args.image.split('/')[-1].split('.')[0]}_{args.goal.split('/')[-1].split('.')[0]}"

    filename = os.path.join(output_dir, filename)

    iio.imwrite(
        f"{filename}_rgb.mp4",
        (np.clip(rgb, 0, 1) * 255).astype(np.uint8),
        fps=12,
    )
    iio.imwrite(
        f"{filename}_disparity.mp4",
        (colorize_depth(disparity) * 255).astype(np.uint8),
        fps=12,
    )

    print("Building GLB scene")
    for frame_idx in range(pointmap.shape[0])[:: args.pointcloud_save_frame_interval]:
        predictions = {
            "world_points": pointmap[frame_idx : frame_idx + 1],
            "images": rgb[frame_idx : frame_idx + 1],
            "depths": 1 / np.clip(disparity[frame_idx : frame_idx + 1], 1e-8, 1e8),
            "camera_poses": poses[frame_idx : frame_idx + 1],
        }
        scene_3d = predictions_to_glb(
            predictions,
            filter_by_frames="all",
            show_cam=True,
            max_depth=args.max_depth,
            rtol=args.rtol,
            frame_rel_idx=float(frame_idx) / pointmap.shape[0],
        )
        scene_3d.export(f"{filename}_pointcloud_frame_{frame_idx}.glb")
    print("GLB Scene built")


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    seed_all(args.seed)

    if args.num_inference_steps is None:
        args.num_inference_steps = 4 if args.task == "reconstruction" else 50

    if args.guidance_scale is None:
        args.guidance_scale = 1.0 if args.task == "reconstruction" else 3.0

    pipeline = build_pipeline(args)

    if args.task == "reconstruction":
        assert args.video is not None, "Video is required for reconstruction task."
        assert args.image is None, "Image is not required for reconstruction task."
        assert args.goal is None, "Goal is not required for reconstruction task."

        video = iio.imread(args.video).astype(np.float32) / 255.0
        image, goal = None, None
    elif args.task == "prediction":
        assert args.image is not None, "Image is required for prediction task."
        assert args.goal is None, "Goal is not required for prediction task."

        image = PIL.Image.open(args.image)
        video, goal = None, None
    elif args.task == "planning":
        assert args.image is not None, "Image is required for planning task."
        assert args.goal is not None, "Goal is required for planning task."

        image = PIL.Image.open(args.image)
        goal = PIL.Image.open(args.goal)

        video = None

    if args.raymap_action is not None:
        raymap = np.load(args.raymap_action)
    else:
        raymap = None

    if args.task != "reconstruction":
        output = pipeline(
            task=args.task,
            image=image,
            video=video,
            goal=goal,
            raymap=raymap,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=args.fps,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            use_dynamic_cfg=args.use_dynamic_cfg,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            return_dict=True,
        )
        if not args.post_reconstruction:
            save_output(
                rgb=output.rgb,
                disparity=output.disparity,
                raymap=output.raymap,
                args=args,
            )
        else:
            recon_output = pipeline(
                task="reconstruction",
                video=output.rgb,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=args.fps,
                num_inference_steps=4,
                guidance_scale=1.0,  # we don't need guidance scale for reconstruction task
                use_dynamic_cfg=False,
                generator=torch.Generator(device=device).manual_seed(args.seed),
            )
            save_output(
                rgb=output.rgb,
                disparity=recon_output.disparity,
                raymap=recon_output.raymap,
                args=args,
            )
    else:
        # for reconstruction task, we have to employ sliding window on long videos
        window_results = []
        window_indices = get_window_starts(
            len(video), args.num_frames, args.sliding_window_stride
        )
        for start_idx in window_indices:
            output = pipeline(
                task=args.task,
                image=None,
                goal=None,
                video=video[start_idx : start_idx + args.num_frames],
                raymap=raymap[start_idx : start_idx + args.num_frames]
                if raymap is not None
                else None,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=args.fps,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=1.0,  # we don't need guidance scale for reconstruction task
                use_dynamic_cfg=False,
                generator=torch.Generator(device=device).manual_seed(args.seed),
            )
            window_results.append(output)

        # merge window results
        (
            merged_rgb,
            merged_disparity,
            merged_poses,
            pointmaps,
        ) = blend_and_merge_window_results(window_results, window_indices, args)
        save_output(
            rgb=merged_rgb,
            disparity=merged_disparity,
            poses=merged_poses,
            pointmap=pointmaps,
            args=args,
        )


if __name__ == "__main__":
    main()
