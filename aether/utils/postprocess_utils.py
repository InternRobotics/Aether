from __future__ import annotations

from typing import Optional

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from plyfile import PlyData, PlyElement


def signed_log1p_inverse(x):
    """
    Computes the inverse of signed_log1p: x = sign(x) * (exp(abs(x)) - 1).

    Args:
        y (torch.Tensor): Input tensor (output of signed_log1p).

    Returns:
        torch.Tensor: Original tensor x.
    """
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    elif isinstance(x, np.ndarray):
        return np.sign(x) * (np.exp(np.abs(x)) - 1)
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")


def colorize_depth(depth, cmap="Spectral"):
    min_d, max_d = (depth[depth > 0]).min(), (depth[depth > 0]).max()
    depth = (max_d - depth) / (max_d - min_d)

    cm = matplotlib.colormaps[cmap]
    depth = depth.clip(0, 1)
    depth = cm(depth, bytes=False)[..., 0:3]
    return depth


def save_ply(pointmap, image, output_file, downsample=20, mask=None):
    _, h, w, _ = pointmap.shape
    image = image[:, :h, :w]
    pointmap = pointmap[:, :h, :w]

    points = pointmap.reshape(-1, 3)  # (H*W, 3)
    colors = image.reshape(-1, 3)  # (H*W, 3)
    if mask is not None:
        points = points[mask.reshape(-1)]
        colors = colors[mask.reshape(-1)]

    indices = np.random.choice(
        colors.shape[0], int(colors.shape[0] / downsample), replace=False
    )
    points = points[indices]
    colors = colors[indices]

    vertices = []
    for p, c in zip(points, colors):
        vertex = (p[0], p[1], p[2], int(c[0]), int(c[1]), int(c[2]))
        vertices.append(vertex)

    vertex_dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertex_array = np.array(vertices, dtype=vertex_dtype)

    ply_element = PlyElement.describe(vertex_array, "vertex")
    PlyData([ply_element], text=True).write(output_file)


def fov_to_focal(fovx, fovy, h, w):
    focal_x = w * 0.5 / np.tan(fovx)
    focal_y = h * 0.5 / np.tan(fovy)
    focal = (focal_x + focal_y) / 2
    return focal


def get_rays(pose, h, w, focal=None, fovx=None, fovy=None):
    import torch.nn.functional as F

    pose = torch.from_numpy(pose).float()
    x, y = torch.meshgrid(
        torch.arange(w),
        torch.arange(h),
        indexing="xy",
    )
    x = x.flatten().unsqueeze(0).repeat(pose.shape[0], 1)
    y = y.flatten().unsqueeze(0).repeat(pose.shape[0], 1)

    cx = w * 0.5
    cy = h * 0.5
    intrinsics, focal = get_intrinsics(pose.shape[0], h, w, fovx, fovy, focal)
    focal = torch.from_numpy(focal).float()
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal.unsqueeze(-1),
                (y - cy + 0.5) / focal.unsqueeze(-1),
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [t, hw, 3]

    pose = pose.to(dtype=camera_dirs.dtype)
    rays_d = camera_dirs @ pose[:, :3, :3].transpose(1, 2)  # [t, hw, 3]

    rays_o = pose[:, :3, 3].unsqueeze(1).expand_as(rays_d)  # [hw, 3]

    rays_o = rays_o.view(pose.shape[0], h, w, 3)
    rays_d = rays_d.view(pose.shape[0], h, w, 3)

    return rays_o.float().numpy(), rays_d.float().numpy(), intrinsics


def get_intrinsics(batch_size, h, w, fovx=None, fovy=None, focal=None):
    if focal is None:
        focal_x = w * 0.5 / np.tan(fovx)
        focal_y = h * 0.5 / np.tan(fovy)
        focal = (focal_x + focal_y) / 2
    cx = w * 0.5
    cy = h * 0.5
    intrinsics = np.zeros((batch_size, 3, 3))
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy
    intrinsics[:, 2, 2] = 1.0

    return intrinsics, focal


def save_pointmap(
    rgb,
    disparity,
    raymap,
    save_file,
    vae_downsample_scale=8,
    camera_pose=None,
    ray_o_scale_inv=1.0,
    max_depth=1e2,
    save_full_pcd_videos=False,
    smooth_camera=False,
    smooth_method="kalman",  # or simple
    **kwargs,
):
    """

    Args:
        rgb (numpy.ndarray): Shape of (t, h, w, 3), range [0, 1]
        disparity (numpy.ndarray): Shape of (t, h, w), range [0, 1]
        raymap (numpy.ndarray): Shape of (t, 6, h // 8, w // 8)
        ray_o_scale_inv (float, optional): A `ray_o` scale constant. Defaults to 10.
    """
    rgb = np.clip(rgb, 0, 1) * 255

    pointmap_dict = postprocess_pointmap(
        disparity,
        raymap,
        vae_downsample_scale,
        camera_pose,
        ray_o_scale_inv=ray_o_scale_inv,
        smooth_camera=smooth_camera,
        smooth_method=smooth_method,
        **kwargs,
    )

    save_ply(
        pointmap_dict["pointmap"],
        rgb,
        save_file,
        mask=(pointmap_dict["depth"] < max_depth),
    )

    if save_full_pcd_videos:
        pcd_dict = {
            "points": pointmap_dict["pointmap"],
            "colors": rgb,
            "intrinsics": pointmap_dict["intrinsics"],
            "poses": pointmap_dict["camera_pose"],
            "depths": pointmap_dict["depth"],
        }
        np.save(save_file.replace(".ply", "_pcd.npy"), pcd_dict)

    return pointmap_dict


def raymap_to_poses(
    raymap, camera_pose=None, ray_o_scale_inv=1.0, return_intrinsics=True
):
    ts = raymap.shape[0]
    if (not return_intrinsics) and (camera_pose is not None):
        return camera_pose, None, None

    raymap[:, 3:] = signed_log1p_inverse(raymap[:, 3:])

    # Extract ray origins and directions
    ray_o = (
        rearrange(raymap[:, 3:], "t c h w -> t h w c") * ray_o_scale_inv
    )  # [T, H, W, C]
    ray_d = rearrange(raymap[:, :3], "t c h w -> t h w c")  # [T, H, W, C]

    # Compute orientation and directions
    orient = ray_o.reshape(ts, -1, 3).mean(axis=1)  # T, 3
    image_orient = (ray_o + ray_d).reshape(ts, -1, 3).mean(axis=1)  # T, 3
    Focal = np.linalg.norm(image_orient - orient, axis=-1)  # T,
    Z_Dir = image_orient - orient  # T, 3

    # Compute the width (W) and field of view (FoV_x)
    W_Left = ray_d[:, :, :1, :].reshape(ts, -1, 3).mean(axis=1)
    W_Right = ray_d[:, :, -1:, :].reshape(ts, -1, 3).mean(axis=1)
    W = W_Right - W_Left
    W_real = (
        np.linalg.norm(np.cross(W, Z_Dir), axis=-1)
        / (raymap.shape[-1] - 1)
        * raymap.shape[-1]
    )
    Fov_x = np.arctan(W_real / (2 * Focal))

    # Compute the height (H) and field of view (FoV_y)
    H_Up = ray_d[:, :1, :, :].reshape(ts, -1, 3).mean(axis=1)
    H_Down = ray_d[:, -1:, :, :].reshape(ts, -1, 3).mean(axis=1)
    H = H_Up - H_Down
    H_real = (
        np.linalg.norm(np.cross(H, Z_Dir), axis=-1)
        / (raymap.shape[-2] - 1)
        * raymap.shape[-2]
    )
    Fov_y = np.arctan(H_real / (2 * Focal))

    # Compute X, Y, and Z directions for the camera
    X_Dir = W_Right - W_Left
    Y_Dir = np.cross(Z_Dir, X_Dir)
    X_Dir = np.cross(Y_Dir, Z_Dir)

    X_Dir /= np.linalg.norm(X_Dir, axis=-1, keepdims=True)
    Y_Dir /= np.linalg.norm(Y_Dir, axis=-1, keepdims=True)
    Z_Dir /= np.linalg.norm(Z_Dir, axis=-1, keepdims=True)

    # Create the camera-to-world (camera_pose) transformation matrix
    if camera_pose is None:
        camera_pose = np.zeros((ts, 4, 4))
        camera_pose[:, :3, 0] = X_Dir
        camera_pose[:, :3, 1] = Y_Dir
        camera_pose[:, :3, 2] = Z_Dir
        camera_pose[:, :3, 3] = orient
        camera_pose[:, 3, 3] = 1.0

    return camera_pose, Fov_x, Fov_y


def postprocess_pointmap(
    disparity,
    raymap,
    vae_downsample_scale=8,
    camera_pose=None,
    focal=None,
    ray_o_scale_inv=1.0,
    smooth_camera=False,
    smooth_method="simple",
    **kwargs,
):
    """

    Args:
        disparity (numpy.ndarray): Shape of (t, h, w), range [0, 1]
        raymap (numpy.ndarray): Shape of (t, 6, h // 8, w // 8)
        ray_o_scale_inv (float, optional): A `ray_o` scale constant. Defaults to 10.
    """
    depth = np.clip(1.0 / np.clip(disparity, 1e-3, 1), 0, 1e8)

    camera_pose, fov_x, fov_y = raymap_to_poses(
        raymap,
        camera_pose=camera_pose,
        ray_o_scale_inv=ray_o_scale_inv,
        return_intrinsics=(focal is not None),
    )
    if focal is None:
        focal = fov_to_focal(
            fov_x,
            fov_y,
            int(raymap.shape[2] * vae_downsample_scale),
            int(raymap.shape[3] * vae_downsample_scale),
        )

    if smooth_camera:
        # Check if sequence is static
        is_static, trans_diff, rot_diff = detect_static_sequence(camera_pose)

        if is_static:
            print(
                f"Detected static/near-static sequence (trans_diff={trans_diff:.6f}, rot_diff={rot_diff:.6f})"
            )
            # Apply stronger smoothing for static sequences
            camera_pose = adaptive_pose_smoothing(camera_pose, trans_diff, rot_diff)
        else:
            if smooth_method == "simple":
                camera_pose = smooth_poses(
                    camera_pose, window_size=5, method="gaussian"
                )
            elif smooth_method == "kalman":
                camera_pose = smooth_trajectory(camera_pose, window_size=5)

    ray_o, ray_d, intrinsics = get_rays(
        camera_pose,
        int(raymap.shape[2] * vae_downsample_scale),
        int(raymap.shape[3] * vae_downsample_scale),
        focal,
    )

    pointmap = depth[..., None] * ray_d + ray_o

    return {
        "pointmap": pointmap,
        "camera_pose": camera_pose,
        "intrinsics": intrinsics,
        "ray_o": ray_o,
        "ray_d": ray_d,
        "depth": depth,
    }


def detect_static_sequence(poses, threshold=0.01):
    """Detect if the camera sequence is static based on pose differences."""
    translations = poses[:, :3, 3]
    rotations = poses[:, :3, :3]

    # Compute translation differences
    trans_diff = np.linalg.norm(translations[1:] - translations[:-1], axis=1).mean()

    # Compute rotation differences (using matrix frobenius norm)
    rot_diff = np.linalg.norm(rotations[1:] - rotations[:-1], axis=(1, 2)).mean()

    return trans_diff < threshold and rot_diff < threshold, trans_diff, rot_diff


def adaptive_pose_smoothing(poses, trans_diff, rot_diff, base_window=5):
    """Apply adaptive smoothing based on motion magnitude."""
    # Increase window size for low motion sequences
    motion_magnitude = trans_diff + rot_diff
    adaptive_window = min(
        41, max(base_window, int(base_window * (0.1 / max(motion_magnitude, 1e-6))))
    )

    # Apply stronger smoothing for low motion
    poses_smooth = smooth_poses(poses, window_size=adaptive_window, method="gaussian")
    return poses_smooth


def get_pixel(H, W):
    # get 2D pixels (u, v) for image_a in cam_a pixel space
    u_a, v_a = np.meshgrid(np.arange(W), np.arange(H))
    # u_a = np.flip(u_a, axis=1)
    # v_a = np.flip(v_a, axis=0)
    pixels_a = np.stack(
        [u_a.flatten() + 0.5, v_a.flatten() + 0.5, np.ones_like(u_a.flatten())], axis=0
    )

    return pixels_a


def project(depth, intrinsic, pose):
    H, W = depth.shape
    pixel = get_pixel(H, W).astype(np.float32)
    points = (np.linalg.inv(intrinsic) @ pixel) * depth.reshape(-1)
    points = pose[:3, :4] @ np.concatenate(
        [points, np.ones((1, points.shape[1]))], axis=0
    )

    points = points.T.reshape(H, W, 3)

    return points


def depth_edge(
    depth: torch.Tensor,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: Optional[torch.Tensor] = None,
) -> torch.BoolTensor:
    """
    Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.

    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    is_numpy = isinstance(depth, np.ndarray)
    if is_numpy:
        depth = torch.from_numpy(depth)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = F.max_pool2d(
            depth, kernel_size, stride=1, padding=kernel_size // 2
        ) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2)
    else:
        diff = F.max_pool2d(
            torch.where(mask, depth, -torch.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ) + F.max_pool2d(
            torch.where(mask, -depth, -torch.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)

    if is_numpy:
        return edge.numpy()
    return edge


@torch.jit.script
def align_rigid(
    p,
    q,
    weights,
):
    """Compute a rigid transformation that, when applied to p, minimizes the weighted
    squared distance between transformed points in p and points in q. See "Least-Squares
    Rigid Motion Using SVD" by Olga Sorkine-Hornung and Michael Rabinovich for more
    details (https://igl.ethz.ch/projects/ARAP/svd_rot.pdf).
    """

    device = p.device
    dtype = p.dtype
    batch, _, _ = p.shape

    # 1. Compute the centroids of both point sets.
    weights_normalized = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    p_centroid = (weights_normalized[..., None] * p).sum(dim=-2)
    q_centroid = (weights_normalized[..., None] * q).sum(dim=-2)

    # 2. Compute the centered vectors.
    p_centered = p - p_centroid[..., None, :]
    q_centered = q - q_centroid[..., None, :]

    # 3. Compute the 3x3 covariance matrix.
    covariance = (q_centered * weights[..., None]).transpose(-1, -2) @ p_centered

    # 4. Compute the singular value decomposition and then the rotation.
    u, _, vt = torch.linalg.svd(covariance)
    s = torch.eye(3, dtype=dtype, device=device)
    s = s.expand((batch, 3, 3)).contiguous()
    s[..., 2, 2] = (u.det() * vt.det()).sign()
    rotation = u @ s @ vt

    # 5. Compute the optimal scale
    scale = (
        (torch.einsum("b i j, b k j -> b k i", rotation, p_centered) * q_centered).sum(
            -1
        )
        * weights
    ).sum(-1) / ((p_centered**2).sum(-1) * weights).sum(-1)
    # scale = (torch.einsum("b i j, b k j -> b k i", rotation, p_centered) * q_centered).sum([-1, -2]) / (p_centered**2).sum([-1, -2])

    # 6. Compute the optimal translation.
    translation = q_centroid - torch.einsum(
        "b i j, b j -> b i", rotation, p_centroid * scale[:, None]
    )

    return rotation, translation, scale


def align_camera_extrinsics(
    cameras_src: torch.Tensor,  # Bx3x4 tensor representing [R | t]
    cameras_tgt: torch.Tensor,  # Bx3x4 tensor representing [R | t]
    estimate_scale: bool = True,
    eps: float = 1e-9,
):
    """
    Align the source camera extrinsics to the target camera extrinsics.
    NOTE Assume OPENCV convention

    Args:
        cameras_src (torch.Tensor): Bx3x4 tensor representing [R | t] for source cameras.
        cameras_tgt (torch.Tensor): Bx3x4 tensor representing [R | t] for target cameras.
        estimate_scale (bool, optional): Whether to estimate the scale factor. Default is True.
        eps (float, optional): Small value to avoid division by zero. Default is 1e-9.

    Returns:
        align_t_R (torch.Tensor): 1x3x3 rotation matrix for alignment.
        align_t_T (torch.Tensor): 1x3 translation vector for alignment.
        align_t_s (float): Scaling factor for alignment.
    """

    R_src = cameras_src[:, :, :3]  # Extracting the rotation matrices from [R | t]
    R_tgt = cameras_tgt[:, :, :3]  # Extracting the rotation matrices from [R | t]

    RRcov = torch.bmm(R_tgt.transpose(2, 1), R_src).mean(0)
    U, _, V = torch.svd(RRcov)
    align_t_R = V @ U.t()

    T_src = cameras_src[:, :, 3]  # Extracting the translation vectors from [R | t]
    T_tgt = cameras_tgt[:, :, 3]  # Extracting the translation vectors from [R | t]

    A = torch.bmm(T_src[:, None], R_src)[:, 0]
    B = torch.bmm(T_tgt[:, None], R_src)[:, 0]

    Amu = A.mean(0, keepdim=True)
    Bmu = B.mean(0, keepdim=True)

    if estimate_scale and A.shape[0] > 1:
        # get the scaling component by matching covariances
        # of centered A and centered B
        Ac = A - Amu
        Bc = B - Bmu
        align_t_s = (Ac * Bc).mean() / (Ac**2).mean().clamp(eps)
    else:
        # set the scale to identity
        align_t_s = 1.0

    # get the translation as the difference between the means of A and B
    align_t_T = Bmu - align_t_s * Amu

    align_t_R = align_t_R[None]
    return align_t_R, align_t_T, align_t_s


def apply_transformation(
    cameras_src: torch.Tensor,  # Bx3x4 tensor representing [R | t]
    align_t_R: torch.Tensor,  # 1x3x3 rotation matrix
    align_t_T: torch.Tensor,  # 1x3 translation vector
    align_t_s: float,  # Scaling factor
    return_extri: bool = True,
) -> torch.Tensor:
    """
    Align and transform the source cameras using the provided rotation, translation, and scaling factors.
    NOTE Assume OPENCV convention

    Args:
        cameras_src (torch.Tensor): Bx3x4 tensor representing [R | t] for source cameras.
        align_t_R (torch.Tensor): 1x3x3 rotation matrix for alignment.
        align_t_T (torch.Tensor): 1x3 translation vector for alignment.
        align_t_s (float): Scaling factor for alignment.

    Returns:
        aligned_R (torch.Tensor): Bx3x3 tensor representing the aligned rotation matrices.
        aligned_T (torch.Tensor): Bx3 tensor representing the aligned translation vectors.
    """

    R_src = cameras_src[:, :, :3]
    T_src = cameras_src[:, :, 3]

    aligned_R = torch.bmm(R_src, align_t_R.expand(R_src.shape[0], 3, 3))

    # Apply the translation alignment to the source translations
    align_t_T_expanded = align_t_T[..., None].repeat(R_src.shape[0], 1, 1)
    transformed_T = torch.bmm(R_src, align_t_T_expanded)[..., 0]
    aligned_T = transformed_T + T_src * align_t_s

    if return_extri:
        extri = torch.cat([aligned_R, aligned_T.unsqueeze(-1)], dim=-1)
        return extri

    return aligned_R, aligned_T


def slerp(q1, q2, t):
    """Spherical Linear Interpolation between quaternions.
    Args:
        q1: (4,) first quaternion
        q2: (4,) second quaternion
        t: float between 0 and 1
    Returns:
        (4,) interpolated quaternion
    """
    # Compute the cosine of the angle between the two vectors
    dot = np.sum(q1 * q2)

    # If the dot product is negative, slerp won't take the shorter path
    # Fix by negating one of the input quaternions
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # Threshold for using linear interpolation instead of spherical
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # If the inputs are too close for comfort, linearly interpolate
        # and normalize the result
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # Compute the angle between the quaternions
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # Compute interpolation factors
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q1) + (s1 * q2)


def interpolate_poses(pose1, pose2, weight):
    """Interpolate between two camera poses with weight.
    Args:
        pose1: (4, 4) first camera pose
        pose2: (4, 4) second camera pose
        weight: float between 0 and 1, weight for pose1 (1-weight for pose2)
    Returns:
        (4, 4) interpolated pose
    """
    from scipy.spatial.transform import Rotation as R

    # Extract rotations and translations
    R1 = R.from_matrix(pose1[:3, :3])
    R2 = R.from_matrix(pose2[:3, :3])
    t1 = pose1[:3, 3]
    t2 = pose2[:3, 3]

    # Get quaternions
    q1 = R1.as_quat()
    q2 = R2.as_quat()

    # Interpolate rotation using our slerp implementation
    q_interp = slerp(q1, q2, 1 - weight)  # 1-weight because weight is for pose1
    R_interp = R.from_quat(q_interp)

    # Linear interpolation for translation
    t_interp = weight * t1 + (1 - weight) * t2

    # Construct interpolated pose
    pose_interp = np.eye(4)
    pose_interp[:3, :3] = R_interp.as_matrix()
    pose_interp[:3, 3] = t_interp

    return pose_interp


def smooth_poses(poses, window_size=5, method="gaussian"):
    """Smooth camera poses temporally.
    Args:
        poses: (N, 4, 4) camera poses
        window_size: int, must be odd number
        method: str, 'gaussian' or 'savgol' or 'ma'
    Returns:
        (N, 4, 4) smoothed poses
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation as R

    assert window_size % 2 == 1, "window_size must be odd"
    N = poses.shape[0]
    smoothed = np.zeros_like(poses)

    # Extract translations and quaternions
    translations = poses[:, :3, 3]
    rotations = R.from_matrix(poses[:, :3, :3])
    quats = rotations.as_quat()  # (N, 4)

    # Ensure consistent quaternion signs to prevent interpolation artifacts
    for i in range(1, N):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    # Smooth translations
    if method == "gaussian":
        sigma = window_size / 6.0  # approximately 99.7% of the weight within the window
        smoothed_trans = gaussian_filter1d(translations, sigma, axis=0, mode="nearest")
        smoothed_quats = gaussian_filter1d(quats, sigma, axis=0, mode="nearest")
    elif method == "savgol":
        # Savitzky-Golay filter: polynomial fitting
        poly_order = min(window_size - 1, 3)
        smoothed_trans = savgol_filter(
            translations, window_size, poly_order, axis=0, mode="nearest"
        )
        smoothed_quats = savgol_filter(
            quats, window_size, poly_order, axis=0, mode="nearest"
        )
    elif method == "ma":
        # Simple moving average
        kernel = np.ones(window_size) / window_size
        smoothed_trans = np.array(
            [np.convolve(translations[:, i], kernel, mode="same") for i in range(3)]
        ).T
        smoothed_quats = np.array(
            [np.convolve(quats[:, i], kernel, mode="same") for i in range(4)]
        ).T

    # Normalize quaternions
    smoothed_quats /= np.linalg.norm(smoothed_quats, axis=1, keepdims=True)

    # Reconstruct poses
    smoothed_rots = R.from_quat(smoothed_quats).as_matrix()

    for i in range(N):
        smoothed[i] = np.eye(4)
        smoothed[i, :3, :3] = smoothed_rots[i]
        smoothed[i, :3, 3] = smoothed_trans[i]

    return smoothed


def smooth_trajectory(poses, window_size=5):
    """Smooth camera trajectory using Kalman filter.
    Args:
        poses: (N, 4, 4) camera poses
        window_size: int, window size for initial smoothing
    Returns:
        (N, 4, 4) smoothed poses
    """
    from filterpy.kalman import KalmanFilter
    from scipy.spatial.transform import Rotation as R

    N = poses.shape[0]

    # Initialize Kalman filter for position and velocity
    kf = KalmanFilter(dim_x=6, dim_z=3)  # 3D position and velocity
    dt = 1.0  # assume uniform time steps

    # State transition matrix
    kf.F = np.array(
        [
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    # Measurement matrix
    kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

    # Measurement noise
    kf.R *= 0.1

    # Process noise
    kf.Q *= 0.1

    # Initial state uncertainty
    kf.P *= 1.0

    # Extract translations and rotations
    translations = poses[:, :3, 3]
    rotations = R.from_matrix(poses[:, :3, :3])
    quats = rotations.as_quat()

    # First pass: simple smoothing for initial estimates
    smoothed = smooth_poses(poses, window_size, method="gaussian")
    smooth_trans = smoothed[:, :3, 3]

    # Second pass: Kalman filter for trajectory
    filtered_trans = np.zeros_like(translations)
    kf.x = np.zeros(6)
    kf.x[:3] = smooth_trans[0]

    filtered_trans[0] = smooth_trans[0]

    # Forward pass
    for i in range(1, N):
        kf.predict()
        kf.update(smooth_trans[i])
        filtered_trans[i] = kf.x[:3]

    # Backward smoothing for rotations using SLERP
    window_half = window_size // 2
    smoothed_quats = np.zeros_like(quats)

    for i in range(N):
        start_idx = max(0, i - window_half)
        end_idx = min(N, i + window_half + 1)
        weights = np.exp(
            -0.5 * ((np.arange(start_idx, end_idx) - i) / (window_half / 2)) ** 2
        )
        weights /= weights.sum()

        # Weighted average of nearby quaternions
        avg_quat = np.zeros(4)
        for j, w in zip(range(start_idx, end_idx), weights):
            if np.dot(quats[j], quats[i]) < 0:
                avg_quat += w * -quats[j]
            else:
                avg_quat += w * quats[j]
        smoothed_quats[i] = avg_quat / np.linalg.norm(avg_quat)

    # Reconstruct final smoothed poses
    final_smoothed = np.zeros_like(poses)
    smoothed_rots = R.from_quat(smoothed_quats).as_matrix()

    for i in range(N):
        final_smoothed[i] = np.eye(4)
        final_smoothed[i, :3, :3] = smoothed_rots[i]
        final_smoothed[i, :3, 3] = filtered_trans[i]

    return final_smoothed


def compute_scale(prediction, target, mask):
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).bool()

    numerator = torch.sum(mask * prediction * target, (1, 2))
    denominator = torch.sum(mask * prediction * prediction, (1, 2))

    scale = torch.zeros_like(numerator)

    valid = (denominator != 0).nonzero()

    scale[valid] = numerator[valid] / denominator[valid]

    return scale.item()
