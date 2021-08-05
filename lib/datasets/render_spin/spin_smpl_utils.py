import math
import torch
import numpy as np
from smplx import SMPL as smpl
from torchgeometry import rotation_matrix_to_angle_axis

SMPL_JOINT_MAPPER = lambda joints: joints[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]

def to_homo(mat):
    row = np.array([[0, 0, 0, 1.]], dtype=np.float32)
    if len(mat.shape) == 3:
        row = row.reshape(1, 1, 4).repeat(len(mat), 0)
    return np.concatenate([mat, row], axis=1)




def rotate_x(phi):
    cos = np.cos(phi)
    sin = np.sin(phi)
    return np.array([[1,   0,    0, 0],
                     [0, cos, -sin, 0],
                     [0, sin,  cos, 0],
                     [0,   0,    0, 1]], dtype=np.float32)

def rotate_z(psi):
    cos = np.cos(psi)
    sin = np.sin(psi)
    return np.array([[cos, -sin, 0, 0],
                     [sin,  cos, 0, 0],
                     [0,      0, 1, 0],
                     [0,      0, 0, 1]], dtype=np.float32)
def rotate_y(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos,   0, -sin, 0],
                     [0,     1,    0, 0],
                     [sin,   0,  cos, 0],
                     [0,   0,      0, 1]], dtype=np.float32)

def generate_bullet_time(c2w, n_views=20, along_z=False):

    y_angles = -np.linspace(0, math.radians(360), n_views+1)[:-1]
    c2ws = []
    for a in y_angles:
        if not along_z:
            c = rotate_y(a) @ c2w
        else:
            c = rotate_z(a) @ c2w
        c2ws.append(c)
    return np.array(c2ws)

def load_smpl_from_paths(smpl_paths):

    Rhs, Ths, poses, shapes =[], [], [], []

    for path in smpl_paths:
        smpl = np.load(path, allow_pickle=True).item()
        Rhs.append(smpl['Rh'])
        Ths.append(smpl['Th'])
        poses.append(smpl['poses'])
        shapes.append(smpl['shapes'])
    Rhs = np.concatenate(Rhs)
    Ths = np.concatenate(Ths)
    poses = np.concatenate(poses)
    shapes = np.concatenate(shapes)

    return Rhs, Ths, poses, shapes

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)

    w = norm_quat[:, 0]
    x = norm_quat[:, 1]
    y = norm_quat[:, 2]
    z = norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ], dim=1).view(batch_size, 3, 3)

    return rotMat

def axisang_to_rot(axisang):
    """
    From https://github.com/gulvarol/smplpytorch/blob/master/smplpytorch/pytorch/rodrigues_layer.py
    """

    angle = torch.norm(axisang + 1e-8, p=2, dim=-1)[..., None]
    axisang_norm = axisang / angle
    angle = angle * 0.5

    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)

    quat = torch.cat([v_cos, v_sin * axisang_norm], dim=-1)
    rot = quat2mat(quat)
    """
    zero_col = torch.zeros(1, 3, 1).expand(rot.size(0), 3, 1)
    ho_row = torch.tensor([[[0., 0., 0., 1.]]]).expand(rot.size(0), 1, 4)
    rot = torch.cat([rot, zero_col], dim=-1)
    rot = torch.cat([rot, ho_row], dim=-2)
    """
    return rot

def rot_to_axisang(rot):
    zeros = torch.zeros(rot.shape[0], 3, 1).to(rot.device)
    rot = torch.cat([rot, zeros], dim=-1)
    return rotation_matrix_to_angle_axis(rot)

@torch.no_grad()
def spin_smpl_to_nb(shapes, poses, gender="NEUTRAL", mapper=SMPL_JOINT_MAPPER):

    #root_rot = torch.FloatTensor(rot_to_zju) @ poses[..., :1, :, :] # (B, 1, 3, 3)
    Rh = poses[..., :1, :]
    Th = torch.zeros(len(shapes), 3)

    with torch.no_grad():
        # remove global transformation for now.
        dummy = torch.eye(3).view(1, 1, 3, 3).expand(len(shapes), 24, 3, 3)
        rots = axisang_to_rot(poses.view(-1, 3)).view(-1, 24, 3, 3)

        body_model = smpl(f"smpl/SMPL_NEUTRAL.pkl",
                         joint_mapper=mapper)
        smpl_output = body_model(betas=shapes,
                             body_pose=rots[..., 1:, :, :],
                             global_orient=rots[..., :1, :, :],#dummy[..., :1, :, :],#root_rot,
                             pose2rot=False,
                             )
        # TODO: remove root translation.
        # note that: Root translation is not in zju coordinate system.
        # Need to first remove t, rotate t to zju system, and then apply it
        # We do not have to do this for c2w, since that when we do R @ c2w,
        # the rotation and transformation will still be applied in the correct order
        # (e.g., rotation happened first, and then translation)
        Th = smpl_output.joints[:, :1]
        vertices = smpl_output.vertices# - Th
        #Th = Th @ torch.FloatTensor(rot_to_zju).T # put translation to zju coordinate
    # ignore root_rot
    N_J = poses.shape[1] - 1
    poses = poses[..., 1:, ...]

    # transform vertices to world coordinate (of ZJU)

    #vertices = torch.einsum("bij,bvj->bvi", root_rot[:, 0], vertices)

    return {"vertices": vertices.numpy(),
            "Rh": Rh.numpy(), "Th": Th.numpy(),
            "shapes": shapes.numpy(), "poses": poses.numpy(), "smpl_model": body_model}

def nerf_bones_to_smpl(bones):
    # undo local transformation
    bones = torch.cat([bones[..., 0:1], -bones[..., 2:3], bones[..., 1:2]], dim=-1)
    rots = axisang_to_rot(bones.view(-1, 3)).view(*bones.shape[:2], 3, 3)
    # undo global transformation
    root_rot = torch.tensor([[1., 0., 0.],
                             [0., 0.,-1.],
                             [0., 1., 0.]]).to(bones.device)
    root_rot = root_rot.expand(len(rots), 3, 3)
    rots[:, 0] = root_rot @ rots[:, 0]
    return rots

