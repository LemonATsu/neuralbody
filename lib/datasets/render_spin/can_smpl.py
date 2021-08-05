import torch
import h5py
import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import deepdish as dd
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
from .render_catalog import RenderCatalog
from .spin_smpl_utils import spin_smpl_to_nb, load_smpl_from_paths, to_homo, generate_bullet_time
from .spin_smpl_utils import rotate_x, rotate_y, nerf_bones_to_smpl, rot_to_axisang

# NOTE: FROM OUR SYSTEM - Y flipped, X off by 90 deg
shape_list = {
    'weipeng': 'data/MonoPerfCap/Weipeng_outdoor/Weipeng_outdoor/smpl/frame_c_0_f_1300.npy',
    'nadia': 'data/MonoPerfCap/Nadia_outdoor/Nadia_outdoor/smpl/frame_1934.npy',
    'S9': 'data/h36m/S9/Directions-1/smpl/54138969/img_002305.npy',
    'S11': 'data/h36m/S11/Directions-1/smpl/54138969/img_001537.npy',
    'archer': 'data/mixamo/Archer/Thriller/Camera_0/smpl/Image0650.npy',
    'james': 'data/mixamo/James/Thriller/Camera_0/smpl/Image0650.npy',
}
def find_idxs_with_map(selected_idxs, idx_map):
    if idx_map is None:
        return selected_idxs
    match_idxs = []
    for sel in selected_idxs:
        for i, m in enumerate(idx_map):
            if m == sel:
                match_idxs.append(i)
                break
    return np.array(match_idxs)

def load_bubble(data_root, annots, smpl_path_map, selected_idxs,
                x_deg=15., y_deg=25., z_t=0.1, n_step=5, refined=None):
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
    T[..., :2, 0] = 0.
    z_t = z_t * T[0,  2, -1]

    # set motions
    motions = np.linspace(0., 2 * np.pi, n_step, endpoint=True)
    x_motions = (np.cos(motions) - 1.) * x_rad
    y_motions = -np.sin(motions) * y_rad
    z_trans = -(np.sin(motions) + 1.) * z_t
    print("NEG TRANS")

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


def load_surreal(annots, selected_idxs, **kwargs):
    import pdb; pdb.set_trace()
    cams = {'K': annots['K'][selected_idxs],
            'R': annots['R'][selected_idxs],
            'T': annots['T'][selected_idxs],
            'cam_inds': selected_idxs}

    N_kps = annots['vertices'].shape[0]
    kp_idxs = selected_idxs % N_kps
    smpls = {'vertices': annots['vertices'][kp_idxs],
             'Rh': annots['Rh'][kp_idxs],
             'Th': annots['Th'][kp_idxs],
             'shapes': annots['shapes'][:],
             'poses': annots['poses'][kp_idxs]}

    """
    n_bullet = 5
    RT = np.concatenate([cams['R'], cams['T']], axis=-1)
    RT_h = to_homo(RT)
    c2ws = generate_bullet_time(np.linalg.inv(RT_h), n_bullet).transpose(1, 0, 2, 3)
    RT = np.linalg.inv(c2ws)
    R = RT[..., :3, :3].reshape(-1, 3, 3)
    T = RT[..., :3, -1:].reshape(-1, 3, 1)
    K = cams['K'][:, None].repeat(n_bullet, 1).reshape(-1, 3, 3)
    selected_idxs = selected_idxs[:, None].repeat(n_bullet, 1).reshape(-1)

    cams = {"K": K, "R": R, "T": T, "cam_inds": selected_idxs}
    """


    return None, smpls, cams


def load_retarget(data_root, annots, smpl_path_map, selected_idxs, length=30, skip=1,
                  center_kps=False, refined=None, idx_map=None):
    l = length
    selected_idxs = np.concatenate([np.arange(s, s+l)[::skip] for s in selected_idxs])

    cams = annots['cams']
    ims = np.array(annots['ims'])[selected_idxs]

    smpl_paths = [os.path.join(data_root, im.replace(*smpl_path_map)[:-4] + '.npy')
                  for im in ims]
    K, R, T = cams['K'][selected_idxs], cams['R'][selected_idxs], cams['T'][selected_idxs]
    cams = {"K": K, "R": R, "T": T, "cam_inds": selected_idxs}
    if refined is not None:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        shapes = refined[1].item()['shapes']
        poses = torch.FloatTensor(refined[0])
        poses = nerf_bones_to_smpl(poses)
        B, J, _, _ = poses.shape
        poses = poses.reshape(-1, 3, 3)
        poses = rot_to_axisang(poses).reshape(B, J, 3)
        poses = poses[selected_idxs]
    else:
        print("!!!load unrefined poses!!!")
        Rhs, Ths, poses, shapes = load_smpl_from_paths(smpl_paths)
        poses = np.concatenate([Rhs[:, None], poses], axis=1)
    smpls = spin_smpl_to_nb(torch.FloatTensor(shapes), torch.FloatTensor(poses))


    return ims, smpls, cams



def load_bullettime(data_root, annots, smpl_path_map, selected_idxs, n_bullet=10,
                    undo_rot=False, center_cam=True, center_kps=True, refined=None):

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
                     center_cam=False, center_kps=False, n_step=10, refined=None):

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
        self.subject = subject

        # for validation
        self.is_val = False
        self.bg_imgs, self.bg_indices = None, None

        if subject != 'surreal':
            annots = np.load(ann_file, allow_pickle=True).item()
        else:
            annots = h5py.File(ann_file, 'r')

        smpl_path_maps = {
            'perfcap': ('images', 'smpl'),
            'mixamo': ('ImageSequence', 'smpl'),
            'h36m': ('imageSequence', 'smpl'),
        }

        self.data_root = catalog[subject]['data_root']

        refined = None
        if cfg.render_refined:
            shape_file_maps = {
                'S9': 'S9',
                'S11': 'S11',
                'James': 'James',
                'Archer': 'Archer',
                'Weipeng_outdoor': 'weipeng',
                'Nadia_outdoor': 'nadia',
            }
            refined = np.load(f'refined_poses/{subject}_poses.npy')
            original_smpl = np.load(shape_list[cfg.original_subject], allow_pickle=True)
            refined = (refined, original_smpl)

        render_args['refined'] = refined

        if split == 'interpolate':
            ims, smpls, cams = load_interpolate(self.data_root, annots,
                                           smpl_path_maps[data_root], **render_args)
        elif split == 'bullet':
            ims, smpls, cams = load_bullettime(self.data_root, annots,
                                           smpl_path_maps[data_root], **render_args)
        elif split == 'retarget':
            if data_root == 'mixamo':
                idx_map = catalog[subject]['idx_map']
            else:
                idx_map=None
            ims, smpls, cams = load_retarget(self.data_root, annots,
                                             smpl_path_maps[data_root],
                                             idx_map=idx_map,**render_args)
        elif split == 'bubble':
            ims, smpls, cams = load_bubble(self.data_root, annots,
                                           smpl_path_maps[data_root], **render_args)
        elif split == 'all':
            ims, smpls, cams = load_retarget(self.data_root, annots,
                                             smpl_path_maps[data_root],
                                             catalog[subject]['idx_map'], length=1)
        elif split == 'val':
            if subject == 'surreal':
                ims, smpls, cams = load_surreal(annots, **render_args)
            else:
                ims, smpls, cams = load_retarget(self.data_root, annots,
                                             smpl_path_maps[data_root],
                                             render_args['selected_idxs'],
                                             center_kps=False, refined=refined,
                                             length=1, skip=1)
                self.bg_imgs = dd.io.load(catalog[subject]['data_h5'], '/bkgds')
                self.bg_indices = dd.io.load(catalog[subject]['data_h5'], '/bkgd_idxs')[cams['cam_inds']]
                self.bg_indices = self.bg_indices.astype(np.int64)
            self.is_val = True

        self._ims = ims
        self.ims = np.arange(len(smpls['vertices']))
        self.smpls = smpls
        self.cams = cams
        self.num_cams = len(self.ims)
        self.cam_inds = cams['cam_inds']
        self.split = 'test'
        self.nrays = cfg.N_rand

        if data_root == 'perfcap':
            if cfg.H == cfg.W:
                self.cams['K'][..., 0, -1] = self.cams['K'][..., 1, -1]


    def prepare_input(self, index):
        # read xyz, normal, color from the ply file
        xyz = self.smpls['vertices'][index] #* self.mul
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        if not self.is_val:
            min_xyz[0] -= 0.10
            max_xyz[0] += 0.10
            min_xyz[1] -= 0.15
            max_xyz[1] += 0.15
            min_xyz[2] -= 0.10
            max_xyz[2] += 0.10
        else:
            min_xyz[0] -= 0.12
            max_xyz[0] += 0.12
            min_xyz[1] -= 0.17
            max_xyz[1] += 0.17
            min_xyz[2] -= 0.12
            max_xyz[2] += 0.12
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
        K = np.array(self.cams['K'][index])
        R = np.array(self.cams['R'][index])
        T = np.array(self.cams['T'][index]) / self.mul

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

        if self.is_val and not self.subject == 'surreal':
            gt_img = imageio.imread(os.path.join(self.data_root, self._ims[index])) / 255.
            ret['gt_img'] = np.array(gt_img).astype(np.float32)
            if self.bg_imgs is not None:
                ret['bg_img'] = self.bg_imgs[self.bg_indices[index]] / 255.

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        i = index #// self.num_cams
        if cfg.selected_framecode >= -1:
            i = cfg.selected_framecode
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
