import torch
from utils3d.camera_utils import *
import kaolin
def prepare_vertices(vertices, faces, camera_proj, camera_rot=None, camera_trans=None,
                    camera_transform=None):
    if camera_transform is None:
        assert camera_trans is not None and camera_rot is not None, \
            "camera_transform or camera_trans and camera_rot must be defined"
        vertices_camera = kaolin.render.camera.rotate_translate_points(vertices, camera_rot,
                                                        camera_trans)
    else:
        assert camera_trans is None and camera_rot is None, \
            "camera_trans and camera_rot must be None when camera_transform is defined"
        padded_vertices = torch.nn.functional.pad(
            vertices, (0, 1), mode='constant', value=1.
        )
        vertices_camera = (padded_vertices @ camera_transform)
    # Project the vertices on the camera image plan
    vertices_image = perspective_camera(vertices_camera, camera_proj)
    face_vertices_camera = kaolin.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
    face_vertices_image = kaolin.ops.mesh.index_vertices_by_faces(vertices_image, faces)
    face_normals = kaolin.ops.mesh.face_normals(face_vertices_camera, unit=True)
    return face_vertices_camera, face_vertices_image, face_normals
    
def render_mesh(verts, faces, intrinsics, extrinsics, resolution, color = None):
    #camera_proj = torch.stack([intrinsics[:, 0, 0] / intrinsics[:, 0, 2], intrinsics[:, 1, 1] / intrinsics[:, 1, 2], torch.ones_like(intrinsics[:, 0, 0])], -1).to(device)
    camera_proj = intrinsics
    camera_transform = extrinsics.permute(0, 2, 1)

    if color is None:
        color = torch.ones_like(verts)
    verts = verts.unsqueeze(0).repeat(intrinsics.shape[0], 1, 1)
    faces = faces
    verts_color = color.unsqueeze(0).repeat(intrinsics.shape[0], 1, 1)
    faces_color = verts_color[:, faces]

    face_vertices_camera, face_vertices_image, face_normals = prepare_vertices(
        verts, faces, camera_proj, camera_transform=camera_transform
    )
    face_vertices_image[:, :, :, 1] = -face_vertices_image[:, :, :, 1]
    #face_vertices_camera[:, :, :, 1:] = -face_vertices_camera[:, :, :, 1:]
    face_normals[:, :, 1:] = -face_normals[:, :, 1:]
    ### Perform Rasterization ###
    # Construct attributes that DI1-R rasterizer will interpolate.
    # the first is the UVS associated to each face
    # the second will make a hard segmentation mask
    face_attributes = [
        faces_color,
        torch.ones((faces_color.shape[0], faces_color.shape[1], 3, 1), device=verts.device),
        face_vertices_camera[:, :, :, 2:],
        face_normals.unsqueeze(-2).repeat(1, 1, 3, 1),
    ]

    # If you have nvdiffrast installed you can change rast_backend to
    # nvdiffrast or nvdiffrast_fwd
    image_features, soft_masks, face_idx = kaolin.render.mesh.dibr_rasterization(
        resolution, resolution, -face_vertices_camera[:, :, :, -1],
        face_vertices_image, face_attributes, face_normals[:, :, -1],
        rast_backend='cuda')

    # image_features is a tuple in composed of the interpolated attributes of face_attributes
    images, masks, depths, normals = image_features
    images = torch.clamp(images * masks, 0., 1.)
    depths = (depths * masks)
    normals = (normals * masks)

    return images, soft_masks, depths, normals
