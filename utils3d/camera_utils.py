import torch

def intrinsic_normalize(intrinsic, ori_reso):
    # intrinsic [B, 3, 3]
    # normalize the intrinsic matrix
    intrinsic[..., 0, 0] = intrinsic[..., 0, 0] * 2 / ori_reso
    intrinsic[..., 1, 1] = intrinsic[..., 1, 1] * 2 / ori_reso
    intrinsic[..., [0, 1], 2] = intrinsic[..., [0, 1], 2] * 2 / ori_reso - 1
    return intrinsic
    

def perspective_camera(points, camera_proj):
    # points [B, N, 3]
    # camera_proj [B, 3, 3]
    # project the 3D points to camera image plane
    projected_points = torch.bmm(points, camera_proj.permute(0, 2, 1))
    projected_2d_points = projected_points[:, :, :2] / projected_points[:, :, 2:3]

    return projected_2d_points

def project_calibrations(query_pts, calibrations):
    query_pts = torch.bmm(calibrations[:, :3, :3], query_pts)
    query_pts = query_pts + calibrations[:, :3, 3:4]
    query_pts_xy = query_pts[:, :2, :] / query_pts[:, 2:, :]
    query_pts_xy = query_pts_xy
    return query_pts_xy


def project_intrinsic_extrinsic(points_3d, intrinsic, extrinsic):
    points_3d = points_3d.permute(0,2,1)
    calibrations = torch.bmm(intrinsic, extrinsic)
    points_2d = project_calibrations(points_3d, calibrations)
    points_2d = points_2d.permute(0,2,1)
    return points_2d