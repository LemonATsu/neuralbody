import torch
import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
from .render_catalog import RenderCatalog
from .spin_smpl_utils import spin_smpl_to_nb, load_smpl_from_paths, to_homo, generate_bullet_time
from .spin_smpl_utils import rotate_x, rotate_y

# NOTE: FROM OUR SYSTEM - Y flipped, X off by 90 deg

def load_bubble(data_root, annots, smpl_path_map, selected_idxs,
                x_deg=15., y_deg=25., z_t=0.1, n_step=5):
    x_rad = x_deg * np.pi / 180.
    y_rad = y_deg * np.pi / 180.

    cams = annots['cams']
    ims = np.array(annots['ims'])[selected_idxs]

    smpl_paths = [os.path.join(data_root, im.replace(*smpl_path_map)[:-4] + '.npy')
                  for im in ims]
    Rhs, Ths, poses, shapes = load_smpl_from_paths(smpl_paths)
    poses = np.concatenate([Rhs[:, None], poses], axis=1)[:, None].repeat(n_step, 1).reshape(-1, 24, 3)
    poses = poses.reshape(-1, 24, 3)
    shapes = shapes[:, None].repeat(n_step, 1).reshape(-1, 10)
    smpls = spin_smpl_to_nb(torch.FloatTensor(shapes), torch.FloatTensor(poses))

    # center pose
    smpls['vertices'] -= smpls['Th'].copy()
    smpls['Th'] *= 0.


    K, R, T = cams['K'][selected_idxs], cams['R'][selected_idxs], cams['T'][selected_idxs]

    # center camera
    shift_x = T[..., 0, -1].copy()
    shift_y = T[..., 1, -1].copy()
    T[..., :2, 0] -= 0.
    z_t = z_t * T[0,  2, -1]

    # set motions
    motions = np.linspace(0., 2 * np.pi, n_step, endpoint=True)
    x_motions = (np.cos(motions) - 1.) * x_rad
    y_motions = -np.sin(motions) * y_rad
    z_trans = (np.sin(motions) + 1.) * z_t

    cam_motions = []
    for x_motion, y_motion in zip(x_motions, y_motions):
        cam_motion = rotate_x(x_motion) @ rotate_y(y_motion)
        cam_motions.append(cam_motion)
    RT = np.concatenate([R, T], axis=-1)
    RT_h = to_homo(RT)
    c2ws = np.linalg.inv(RT_h)

    # apply motion to camera
    bubble_c2ws = []
    for c2w in c2ws:
        bubbles = []
        for cam_motion, z_tran in zip(cam_motions, z_trans):
            c = c2w.copy()
            c[2, -1] += z_tran
            bubbles.append(cam_motion @ c)
        bubble_c2ws.append(bubbles)
    bubble_c2ws = np.array(bubble_c2ws).reshape(-1, 4, 4)
    bubble_RT = np.linalg.inv(bubble_c2ws)

    R = bubble_RT[..., :3, :3]
    T = bubble_RT[..., :3, -1:]
    K = K[:, None].repeat(n_step, 1).reshape(-1, 3, 3)
    selected_idxs = selected_idxs[:, None].repeat(n_step, 1).reshape(-1)

    cams = {"K": K, "R": R, "T": T, "cam_inds": selected_idxs}
    return ims, smpls, cams


def load_retarget(data_root, annots, smpl_path_map, selected_idxs, length=30, skip=1,
                  center_kps=False):
    l = length
    selected_idxs = np.concatenate([np.arange(s, s+l)[::skip] for s in selected_idxs])

    cams = annots['cams']
    ims = np.array(annots['ims'])[selected_idxs]

    smpl_paths = [os.path.join(data_root, im.replace(*smpl_path_map)[:-4] + '.npy')
                  for im in ims]
    Rhs, Ths, poses, shapes = load_smpl_from_paths(smpl_paths)
    poses = np.concatenate([Rhs[:, None], poses], axis=1)
    smpls = spin_smpl_to_nb(torch.FloatTensor(shapes), torch.FloatTensor(poses))

    K, R, T = cams['K'][selected_idxs], cams['R'][selected_idxs], cams['T'][selected_idxs]
    cams = {"K": K, "R": R, "T": T, "cam_inds": selected_idxs}

    return ims, smpls, cams



def load_bullettime(data_root, annots, smpl_path_map, selected_idxs, n_bullet=30,
                    undo_rot=False, center_cam=True, center_kps=True):

    cams = annots['cams']
    ims = np.array(annots['ims'])[selected_idxs]

    # load and create vertice
    smpl_paths = [os.path.join(data_root, im.replace(*smpl_path_map)[:-4] + '.npy')
                  for im in ims]
    Rhs, Ths, poses, shapes = load_smpl_from_paths(smpl_paths)
    if undo_rot:
        Rhs[:, :] = np.array([1.5708*2, 0., 0.], dtype=np.float32).reshape(1, 3)
    poses = np.concatenate([Rhs[:, None], poses], axis=1)
    smpls = spin_smpl_to_nb(torch.FloatTensor(shapes), torch.FloatTensor(poses))

    # set up camera setting
    K, R, T = cams['K'][selected_idxs], cams['R'][selected_idxs], cams['T'][selected_idxs]
    if center_cam:
        shift_x = T[..., 0, -1].copy()
        shift_y = T[..., 1, -1].copy()
        T[..., :2, 0] = 0.

        print('CAM CENTERED')

    # rotate camera
    RT = np.concatenate([R, T], axis=-1)
    RT_h = to_homo(RT)
    c2ws = generate_bullet_time(np.linalg.inv(RT_h), n_bullet).transpose(1, 0, 2, 3)
    RT = np.linalg.inv(c2ws)
    R = RT[..., :3, :3].reshape(-1, 3, 3)
    T = RT[..., :3, -1:].reshape(-1, 3, 1)
    K = K[:, None].repeat(n_bullet, 1).reshape(-1, 3, 3)
    selected_idxs = selected_idxs[:, None].repeat(n_bullet, 1).reshape(-1)

    cams = {"K": K, "R": R, "T": T, "cam_inds": selected_idxs}

    if center_kps:
        smpls['vertices'] -= smpls['Th'].copy()
        smpls['Th'] *= 0
        print('VERT CENTERED')
    elif center_cam:
        smpls['vertices'][..., 0] -= shift_x[:, None]
        smpls['vertices'][..., 1] -= shift_y[:, None]
        smpls['Th'][..., 0] -= shift_x[:, None]
        smpls['Th'][..., 1] -= shift_y[:, None]

    for k in smpls:
        if k != 'smpl_model':
            sh = smpls[k].shape
            smpls[k] = smpls[k][:, None].repeat(n_bullet, 1).reshape(-1, *sh[1:])
    return ims, smpls, cams

def load_interpolate(data_root, annots, smpl_path_map, selected_idxs, undo_rot=False,
                     center_cam=False, center_kps=False, n_step=10):

    cams = annots['cams']
    # load and create vertice
    ims = np.array(annots['ims'])[selected_idxs]
    print(ims)
    smpl_paths = [os.path.join(data_root, im.replace(*smpl_path_map)[:-4] + '.npy')
                  for im in ims]
    Rhs, Ths, poses, shapes = load_smpl_from_paths(smpl_paths)
    if undo_rot:
        Rhs[:, :] = np.array([1.5708*2, 0., 0.], dtype=np.float32).reshape(1, 3)
    poses = np.concatenate([Rhs[:, None], poses], axis=1)

    interp_poses, interp_shapes = [], []
    w = np.linspace(0, 1.0, n_step, endpoint=False, dtype=np.float32).reshape(-1, 1, 1)
    for i in range(len(poses)-1):
        pose = poses[i:i+1]
        next_pose = poses[i+1:i+2]

        shape = shapes[i:i+1]
        next_shape = shapes[i+1:i+2]

        interp_pose = pose * (1 - w) + next_pose * w
        interp_shape = shape * (1 - w[:, 0]) + next_shape * w[:, 0]

        interp_poses.append(interp_pose)
        interp_shapes.append(interp_shape)

    interp_poses.append(poses[-1:])
    interp_shapes.append(shapes[-1:])

    interp_poses = np.concatenate(interp_poses, axis=0).astype(np.float32)
    interp_shapes = np.concatenate(interp_shapes, axis=0).astype(np.float32)
    interp_smpls = spin_smpl_to_nb(torch.FloatTensor(interp_shapes), torch.FloatTensor(interp_poses))

    interp_smpls.pop('smpl_model') # don't need this
    vertices = interp_smpls['vertices']

    K, R, T = cams['K'][selected_idxs], cams['R'][selected_idxs], cams['T'][selected_idxs]
    K = K[:1].repeat(len(vertices), 0)
    R = R[:1].repeat(len(vertices), 0)
    T = T[:1].repeat(len(vertices), 0)
    selected_idxs = np.array(selected_idxs[:1]).repeat(len(vertices), 0)

    cams = {"K": K, "R": R, "T": T, "cam_inds": selected_idxs}

    if center_cam:
        shift_x = T[..., 0, -1].copy()
        shift_y = T[..., 1, -1].copy()
        T[..., :2, 0] = 0.

    if center_kps:
        interp_smpls['vertices'] -= interp_smpls['Th'].copy()
        interp_smpls['Th'] *= 0.

    elif center_cam:
        interp_smpls['vertices'][..., 0] -= shift_x[:, None]
        interp_smpls['vertices'][..., 1] -= shift_y[:, None]
        interp_smpls['Th'][..., 0] -= shift_x[:, None]
        interp_smpls['Th'][..., 1] -= shift_y[:, None]

    return ims, interp_smpls, cams


class Dataset(data.Dataset):

    def __init__(self, data_root, subject, split, center_kps=False,
                 n_bullet=10):
        '''
        data_root: data_h5 in our own code
        subject: entry in our own code
        split: render type in our own code
        '''
        catalog = RenderCatalog[data_root]
        ann_file = catalog[subject]['ann_file']
        render_args = catalog[subject][split]
        idxs = render_args['selected_idxs']
        self.mul = 1.05 if data_root == 'perfcap' else 1.00

        annots = np.load(ann_file, allow_pickle=True).item()

        smpl_path_maps = {
            'perfcap': ('images', 'smpl'),
            'mixamo': ('ImageSequence', 'smpl'),
        }

        self.data_root = catalog[subject]['data_root']

        if split == 'interpolate':
            ims, smpls, cams = load_interpolate(self.data_root, annots,
                                           smpl_path_maps[data_root], **render_args)
        elif split == 'bullet':
            ims, smpls, cams = load_bullettime(self.data_root, annots,
                                           smpl_path_maps[data_root], **render_args)
        elif split == 'retarget':
            ims, smpls, cams = load_retarget(self.data_root, annots,
                                           smpl_path_maps[data_root], **render_args)
        elif split == 'bubble':
            ims, smpls, cams = load_bubble(self.data_root, annots,
                                           smpl_path_maps[data_root], **render_args)

        self._ims = ims
        self.ims = np.arange(len(smpls['vertices']))
        self.smpls = smpls
        self.cams = cams
        self.num_cams = len(self.ims)
        self.cam_inds = cams['cam_inds']
        self.split = 'test'
        self.nrays = cfg.N_rand


    def prepare_input(self, index):
        # read xyz, normal, color from the ply file
        xyz = self.smpls['vertices'][index] * self.mul
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        min_xyz[0] -= 0.10
        max_xyz[0] += 0.10
        min_xyz[1] -= 0.15
        max_xyz[1] += 0.15
        min_xyz[2] -= 0.10
        max_xyz[2] += 0.10
        """
        if cfg.big_box:
            min_xyz -= 0.18
            max_xyz += 0.18
        else:
            min_xyz[2] -= 0.18
            max_xyz[2] += 0.18
        """
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        #params_path = os.path.join(self.data_root,
        #                           self.ims[index].replace("images", "smpl")[:-4] + ".npy")
        Rh = self.smpls['Rh'][index:index+1]
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        #Th = params['Th'].astype(np.float32)
        Th = self.smpls['Th'][index:index+1].astype(np.float32)[0]
        xyz = np.dot(xyz - Th, R)

        # transformation augmentation
        xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.15
            max_xyz += 0.15
        else:
            min_xyz[2] -= 0.15
            max_xyz[2] += 0.15
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        nxyz = nxyz.astype(np.float32)
        feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans

    def __getitem__(self, index):

        msk = np.ones((cfg.H, cfg.W), dtype=np.uint8)

        cam_ind = self.cam_inds[index]
        #img = imageio.imread(os.path.join(self.data_root, self._ims[cam_ind])) / 255.
        K = np.array(self.cams['K'][index])
        R = np.array(self.cams['R'][index])
        T = np.array(self.cams['T'][index])

        # reduce the image resolution by ratio
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        msk = np.ones((H, W), dtype=np.uint8)
        img = np.ones((H, W, 3), dtype=np.uint8)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        #msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
        K[:2] = K[:2] * cfg.ratio

        feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans = self.prepare_input(
            index)

        if cfg.sample_smpl:
            depth_path = os.path.join(self.data_root, 'depth',
                                      self.ims[index])[:-4] + '.npy'
            depth = np.load(depth_path)
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_smpl_ray(
                img, msk, depth, K, R, T, self.nrays, self.split)
        elif cfg.sample_grid:
            # print('sample_grid')
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_grid(
                img, msk, K, R, T, can_bounds, self.nrays, self.split)
        else:
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_h36m(
                img, msk, K, R, T, can_bounds, self.nrays, self.split)
        acc = if_nerf_dutils.get_acc(coord_, msk)

        ret = {
            'feature': feature,
            'coord': coord,
            'out_sh': out_sh,
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'acc': acc,
            'mask_at_box': mask_at_box,
            'index': index,
            #'img_gt': img,
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        i = index #// self.num_cams
        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'center': center,
            'rot': rot,
            'trans': trans,
            'i': i,
            'cam_ind': cam_ind
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
