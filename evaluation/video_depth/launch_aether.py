import argparse
import math
import os

import cv2
import imageio.v3 as iio
import numpy as np
import rootutils
import torch
from accelerate import Accelerator, PartialState
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from aether.pipelines.aetherv1_pipeline_cogvideox import (  # noqa: E402
    AetherV1PipelineCogVideoX,
)

# noqa: E402
from aether.utils.postprocess_utils import (  # noqa: E402
    colorize_depth,
    compute_scale,
)
from evaluation.video_depth.metadata import dataset_metadata  # noqa: E402


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )
    parser.add_argument(
        "--num_inference_step",
        type=int,
        default=4,
        help="number of inference steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="random seed",
    )
    return parser


def process_with_sliding_window(
    pipeline, obs_image, num_inference_step, total_frames, seed
):
    b, t, h, w, c = obs_image.shape
    assert b == 1, "Only batch size 1 is supported"

    max_frames_per_window = 41
    while max_frames_per_window > total_frames:
        max_frames_per_window -= 8
    temporal_stride = 8

    target_h, target_w = 480, 720
    spatial_overlap_h = 60
    spatial_overlap_w = 90

    h_windows = (
        1
        if h <= target_h
        else math.ceil((h - target_h) / (target_h - spatial_overlap_h)) + 1
    )
    w_windows = (
        1
        if w <= target_w
        else math.ceil((w - target_w) / (target_w - spatial_overlap_w)) + 1
    )
    assert h_windows == 1 or w_windows == 1, (h_windows, w_windows)

    spatial_stride_h = (h - target_h) // (h_windows - 1) if h_windows > 1 else 0
    spatial_stride_w = (w - target_w) // (w_windows - 1) if w_windows > 1 else 0

    temporal_windows_rgb, temporal_windows_disparity, temporal_windows_ranges = (
        [],
        [],
        [],
    )

    t_starts = list(range(0, t - max_frames_per_window, temporal_stride))
    t_starts.append(t - max_frames_per_window)  # Adjust last t_start to fit max window

    for t_start in t_starts:
        t_end = min(t_start + max_frames_per_window, t)
        if t_end < t and t_end - t_start < max_frames_per_window:
            t_start = max(0, t - max_frames_per_window)
            t_end = t

        spatial_windows_rgb, spatial_windows_disparity, spatial_windows_ranges = (
            [],
            [],
            [],
        )
        num_windows, stride, is_horizontal = (
            (w_windows, spatial_stride_w, True)
            if w_windows > 1
            else (h_windows, spatial_stride_h, False)
        )

        for i in range(num_windows):
            if is_horizontal:
                h_start, h_end = 0, target_h
                w_start = int(i * stride)
                w_end = w_start + target_w
                if w_end > w:
                    w_start, w_end = w - target_w, w
            else:
                w_start, w_end = 0, target_w
                h_start = int(i * stride)
                h_end = h_start + target_h
                if h_end > h:
                    h_start, h_end = h - target_h, h

            rgb_videos, disparity_videos, _ = pipeline(
                video=obs_image[0, t_start:t_end, h_start:h_end, w_start:w_end, :],
                num_inference_steps=num_inference_step,
                num_frames=t_end - t_start,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                return_dict=False,
                fps=12,
            )

            spatial_windows_rgb.append(rgb_videos[0])
            spatial_windows_disparity.append(disparity_videos[0])
            spatial_windows_ranges.append(
                (w_start, w_end) if is_horizontal else (h_start, h_end)
            )

        final_spatial_rgb, final_spatial_disparity = None, None
        w1 = spatial_windows_disparity[0]

        for idx, (window_rgb, window_disparity, ranges) in enumerate(
            zip(spatial_windows_rgb, spatial_windows_disparity, spatial_windows_ranges)
        ):
            if idx == 0:
                final_spatial_rgb, final_spatial_disparity = (
                    window_rgb,
                    window_disparity,
                )
            else:
                overlap = spatial_windows_ranges[idx - 1][1] - ranges[0]
                reshape_dims = (1, -1, overlap) if is_horizontal else (1, overlap, -1)

                scale = compute_scale(
                    (
                        window_disparity[:, :, :overlap].reshape(*reshape_dims)
                        if is_horizontal
                        else window_disparity[:, :overlap, :].reshape(*reshape_dims)
                    ),
                    (
                        final_spatial_disparity[:, :, -overlap:].reshape(*reshape_dims)
                        if is_horizontal
                        else final_spatial_disparity[:, -overlap:, :].reshape(
                            *reshape_dims
                        )
                    ),
                    (
                        np.ones_like(final_spatial_disparity[:, :, -overlap:]).reshape(
                            *reshape_dims
                        )
                        if is_horizontal
                        else np.ones_like(
                            final_spatial_disparity[:, -overlap:, :]
                        ).reshape(*reshape_dims)
                    ),
                )
                window_disparity_aligned = scale * window_disparity

                result_shape = (
                    (*w1.shape[:-1], ranges[1])
                    if is_horizontal
                    else (w1.shape[0], ranges[1], w1.shape[2])
                )
                result = np.ones(result_shape)
                weight_shape = (
                    (None, None, slice(None))
                    if is_horizontal
                    else (None, slice(None), None)
                )
                weight = np.linspace(1, 0, overlap)[weight_shape]

                if is_horizontal:
                    result[:, :, : ranges[0]] = final_spatial_disparity[
                        :, :, : ranges[0]
                    ]
                    result[
                        :, :, spatial_windows_ranges[idx - 1][1] :
                    ] = window_disparity_aligned[
                        :, :, spatial_windows_ranges[idx - 1][1] - ranges[0] :
                    ]
                    result[:, :, ranges[0] : spatial_windows_ranges[idx - 1][1]] = (
                        final_spatial_disparity[
                            :, :, ranges[0] : spatial_windows_ranges[idx - 1][1]
                        ]
                        * weight
                        + window_disparity_aligned[:, :, :overlap] * (1 - weight)
                    )
                else:
                    result[:, : ranges[0], :] = final_spatial_disparity[
                        :, : ranges[0], :
                    ]
                    result[
                        :, spatial_windows_ranges[idx - 1][1] :, :
                    ] = window_disparity_aligned[
                        :, spatial_windows_ranges[idx - 1][1] - ranges[0] :, :
                    ]
                    result[:, ranges[0] : spatial_windows_ranges[idx - 1][1], :] = (
                        final_spatial_disparity[
                            :, ranges[0] : spatial_windows_ranges[idx - 1][1], :
                        ]
                        * weight
                        + window_disparity_aligned[:, :overlap, :] * (1 - weight)
                    )

                final_spatial_disparity = result
                # (Additional logic for final_spatial_rgb can be similarly refactored)

        temporal_windows_rgb.append(final_spatial_rgb)
        temporal_windows_disparity.append(final_spatial_disparity)
        temporal_windows_ranges.append((t_start, t_end))

    final_rgb, final_disparity = None, None
    w1 = temporal_windows_disparity[0]

    for idx, (window_rgb, window_disparity, (t_start, t_end)) in enumerate(
        zip(temporal_windows_rgb, temporal_windows_disparity, temporal_windows_ranges)
    ):
        if idx == 0:
            final_rgb, final_disparity = window_rgb, window_disparity
        else:
            overlap_t = temporal_windows_ranges[idx - 1][1] - t_start
            scale = compute_scale(
                window_disparity[:overlap_t].reshape(1, -1, w1.shape[-1]),
                final_disparity[-overlap_t:].reshape(1, -1, w1.shape[-1]),
                np.ones_like(final_disparity[-overlap_t:]).reshape(1, -1, w1.shape[-1]),
            )
            window_disparity_aligned = scale * window_disparity

            result = np.ones((t_end, *w1.shape[1:]))
            result[:t_start] = final_disparity[:t_start]
            result[temporal_windows_ranges[idx - 1][1] :] = window_disparity_aligned[
                temporal_windows_ranges[idx - 1][1] - t_start :
            ]
            weight = np.linspace(1, 0, overlap_t)[:, None, None]
            result[t_start : temporal_windows_ranges[idx - 1][1]] = final_disparity[
                t_start : temporal_windows_ranges[idx - 1][1]
            ] * weight + window_disparity_aligned[:overlap_t] * (1 - weight)
            final_disparity = result

    return final_rgb, final_disparity


def eval_pose_estimation(args, pipeline, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, pipeline, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args, pipeline, img_path, save_dir=None, mask_path=None):
    metadata = dataset_metadata.get(args.eval_dataset)

    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))
            ]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    distributed_state = PartialState()
    pipeline.to(distributed_state.device)

    with distributed_state.split_between_processes(seq_list) as seqs:
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                filelist = [
                    os.path.join(dir_path, name) for name in os.listdir(dir_path)
                ]
                filelist.sort()
                filelist = filelist[:: args.pose_eval_stride]

                obs_image = prepare_input(filelist)
                obs_images = obs_image[None]
                rgb_videos, disparity_videos = process_with_sliding_window(
                    pipeline,
                    obs_images,
                    num_inference_step=args.num_inference_step,
                    total_frames=len(filelist),
                    seed=args.seed,
                )
                disparity_video = disparity_videos
                depth_maps = np.clip(1.0 / disparity_video, 0, 1e2)

                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                path = f"{save_dir}/{seq}"

                iio.imwrite(
                    os.path.join(path, "pred_disparity.mp4"),
                    (colorize_depth(disparity_video) * 255).astype(np.uint8),
                    fps=24,
                )
                iio.imwrite(
                    os.path.join(path, "pred_rgb.mp4"),
                    (np.clip(rgb_videos, 0, 1) * 255).astype(np.uint8),
                    fps=24,
                )
                for i, depth_map in enumerate(depth_maps):
                    np.save(f"{path}/frame_{(i):04d}.npy", depth_map)

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception
    return None, None, None


def prepare_input(img_paths):
    images = []
    for img_path in img_paths:
        img = iio.imread(img_path)
        h, w = img.shape[:2]
        aspect_ratio = w / h

        new_h, new_w = (
            (480, int(round(480 * aspect_ratio)))
            if aspect_ratio > 720 / 480
            else (int(round(720 / aspect_ratio)), 720)
        )

        img = cv2.resize(img, (new_w, new_h)) / 255.0
        images.append(img)
    return np.stack(images)


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    if args.eval_dataset == "sintel":
        args.full_seq = True
    else:
        args.full_seq = False

    accelerator = Accelerator(mixed_precision="bf16")
    pipeline = AetherV1PipelineCogVideoX(
        tokenizer=AutoTokenizer.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            subfolder="tokenizer",
        ),
        text_encoder=T5EncoderModel.from_pretrained(
            "THUDM/CogVideoX-5b-I2V", subfolder="text_encoder"
        ),
        vae=AutoencoderKLCogVideoX.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ),
        scheduler=CogVideoXDPMScheduler.from_pretrained(
            "THUDM/CogVideoX-5b-I2V", subfolder="scheduler"
        ),
        transformer=CogVideoXTransformer3DModel.from_pretrained(
            "AetherWorldModel/AetherV1",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ),
    )
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()

    eval_pose_estimation(args, pipeline, save_dir=args.output_dir)
