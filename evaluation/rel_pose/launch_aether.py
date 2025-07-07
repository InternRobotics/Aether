import argparse
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
from aether.utils.postprocess_utils import (  # noqa: E402
    align_camera_extrinsics,
    apply_transformation,
    compute_scale,
    interpolate_poses,
    postprocess_pointmap,
    smooth_trajectory,
)
from evaluation.rel_pose.evo_utils import (  # noqa: E402
    calculate_averages,
    eval_metrics,
    load_traj,
    plot_trajectory,
    process_directory,
)
from evaluation.rel_pose.metadata import dataset_metadata  # noqa: E402
from evaluation.rel_pose.utils import (  # noqa: E402
    get_tum_poses,
    save_focals,
    save_tum_poses,
)


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
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

        new_w = int(round(new_w / 16) * 16)
        new_h = int(round(new_h / 16) * 16)
        img = cv2.resize(img, (new_w, new_h)) / 255.0

        # center crop
        h, w = img.shape[:2]
        start_h, start_w = (h - 480) // 2, (w - 720) // 2
        img = img[start_h : start_h + 480, start_w : start_w + 720]

        images.append(img)
    return np.stack(images)


def process_video_with_sliding_window(
    pipeline, video_frames, num_inference_steps, seed
):
    """Processes video frames using a temporal sliding window to estimate camera poses."""
    t = video_frames.shape[1]
    max_frames_per_window = 41
    while max_frames_per_window > t:
        max_frames_per_window -= 8
    temporal_stride = 32

    # Generate window start indices
    t_starts = list(range(0, t - max_frames_per_window, temporal_stride))
    if not t_starts or t_starts[-1] != t - max_frames_per_window:
        t_starts.append(t - max_frames_per_window)

    window_outputs = []
    for t_start in t_starts:
        t_end = t_start + max_frames_per_window
        rgb, disparity, raymaps = pipeline(
            video=video_frames[0, t_start:t_end],
            num_inference_steps=num_inference_steps,
            num_frames=t_end - t_start,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            return_dict=False,
            fps=12,
        )

        pcd_dict = postprocess_pointmap(
            disparity[0],
            raymaps[0],
            smooth_camera=True,
            smooth_method="kalman",
        )
        poses = pcd_dict["camera_pose"][:, :3, :4]
        focals = (pcd_dict["intrinsics"][:, 0, 0] + pcd_dict["intrinsics"][:, 1, 1]) / 2
        window_outputs.append(
            {
                "rgb": rgb[0],
                "disparity": disparity[0],
                "poses": poses,
                "focals": focals,
                "range": (t_start, t_end),
            }
        )

    return blend_window_outputs(window_outputs)


def blend_window_outputs(window_outputs):
    """Blends overlapping results from the sliding window."""
    final_results = window_outputs[0]

    for i in range(1, len(window_outputs)):
        prev, curr = final_results, window_outputs[i]
        t_start_curr, _ = curr["range"]
        _, t_end_prev = prev["range"]
        overlap_t = t_end_prev - t_start_curr

        # Align and blend disparity
        # breakpoint()
        scale = compute_scale(
            curr["disparity"][:overlap_t].reshape(1, -1, curr["disparity"].shape[-1]),
            prev["disparity"][-overlap_t:].reshape(
                1, -1, prev["disparity"][-overlap_t:].shape[-1]
            ),
            1.0,
        )
        curr["disparity"] *= scale

        # Align poses
        rel_r, rel_t, rel_s = align_camera_extrinsics(
            torch.from_numpy(curr["poses"][:overlap_t]),
            torch.from_numpy(prev["poses"][-overlap_t:]),
        )
        aligned_poses = (
            apply_transformation(
                torch.from_numpy(curr["poses"]), rel_r, rel_t, rel_s, return_extri=True
            )
            .cpu()
            .numpy()
        )

        # Interpolate poses in overlap region
        weights = np.linspace(1, 0, overlap_t)
        interpolated_poses = np.array(
            [
                interpolate_poses(prev["poses"][t_start_curr + t], aligned_poses[t], w)[
                    :3, :4
                ]
                for t, w in enumerate(weights)
            ]
        )

        # Update final results by appending non-overlapping parts and blended overlap
        for key in ["rgb", "disparity", "poses", "focals"]:
            stitching_point = prev[key].shape[0] - overlap_t
            if key == "poses":
                blended_section = interpolated_poses
                new_section = aligned_poses[overlap_t:]
            else:
                weight_shape = (
                    (overlap_t, 1, 1, 1)
                    if key == "rgb"
                    else (overlap_t, 1, 1)
                    if key != "focals"
                    else (overlap_t,)
                )
                weight = np.linspace(1, 0, overlap_t).reshape(weight_shape)
                blended_section = prev[key][-overlap_t:] * weight + curr[key][
                    :overlap_t
                ] * (1 - weight)
                new_section = curr[key][overlap_t:]

            final_results[key] = np.concatenate(
                (prev[key][:stitching_point], blended_section, new_section), axis=0
            )
        final_results["range"] = (prev["range"][0], curr["range"][-1])

    # Final pose processing
    poses_4x4 = np.concatenate(
        [final_results["poses"], np.zeros((final_results["poses"].shape[0], 1, 4))],
        axis=1,
    )
    poses_4x4[:, -1, 3] = 1.0
    final_results["poses"] = smooth_trajectory(poses_4x4, window_size=5)

    return final_results


def eval_pose_estimation(args, pipeline):
    """Main function to orchestrate the pose estimation evaluation."""
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    anno_path = metadata.get("anno_path")
    save_dir = args.output_dir

    if not args.seq_list:
        if metadata.get("full_seq", False) or args.full_seq:
            args.seq_list = sorted(
                [
                    d
                    for d in os.listdir(img_path)
                    if os.path.isdir(os.path.join(img_path, d))
                ]
            )
        else:
            args.seq_list = metadata.get("seq_list", [])

    distributed_state = PartialState()
    pipeline.to(distributed_state.device)

    with distributed_state.split_between_processes(args.seq_list) as seqs:
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"
        for seq in tqdm(seqs, desc=f"Process {distributed_state.process_index}"):
            try:
                seq_dir = os.path.join(save_dir, seq)
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                dir_path = metadata["dir_path_func"](img_path, seq)
                filelist = sorted(os.listdir(dir_path))[:: args.pose_eval_stride]
                filelist = [os.path.join(dir_path, f) for f in filelist]

                obs_image = prepare_input(filelist)[None]
                results = process_video_with_sliding_window(
                    pipeline, obs_image, args.num_inference_step, seed=42
                )

                os.makedirs(seq_dir, exist_ok=True)
                pred_traj_path = os.path.join(seq_dir, "pred_traj.txt")
                save_tum_poses(results["poses"], pred_traj_path)
                save_focals(
                    {"focal": results["focals"]},
                    os.path.join(seq_dir, "pred_focal.txt"),
                )

                gt_traj_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                traj_format = metadata.get("traj_format", None)

                if args.eval_dataset == "sintel":
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file, stride=args.pose_eval_stride
                    )
                elif traj_format is not None:
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file,
                        stride=args.pose_eval_stride,
                        traj_format=traj_format,
                    )
                else:
                    gt_traj = None

                if gt_traj is not None:
                    ate, rpe_trans, rpe_rot = eval_metrics(
                        get_tum_poses(results["poses"]),
                        gt_traj,
                        seq=seq,
                        filename=os.path.join(seq_dir, "eval_metric.txt"),
                    )
                    plot_trajectory(
                        get_tum_poses(results["poses"]),
                        gt_traj,
                        title=seq,
                        filename=os.path.join(seq_dir, "trajectory.png"),
                    )

                    with open(error_log_path, "a") as f:
                        f.write(
                            f"{args.eval_dataset}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
                        )
                else:
                    ate, rpe_trans, rpe_rot = 0, 0, 0

            except Exception as e:
                with open(error_log_path, "a") as f:
                    f.write(f"ERROR in sequence {seq}: {e}\n")
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print(f"OOM error in sequence {seq}, skipping.")
                else:
                    print(f"An error occurred in sequence {seq}: {e}")
                raise e

    distributed_state.wait_for_everyone()

    if distributed_state.is_main_process:
        results = process_directory(save_dir)
        avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)
        print(
            f"Average ATE: {avg_ate:.5f}, RPE trans: {avg_rpe_trans:.5f}, RPE rot: {avg_rpe_rot:.5f}"
        )


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    args.full_seq = False
    args.no_crop = False

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

    eval_pose_estimation(args, pipeline)
