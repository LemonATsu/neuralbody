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
import h5py


class Dataset(data.Dataset):
    def __init__(self, data_root, split):
        super(Dataset, self).__init__()

        if split == 'train':
            data_root = os.path.join(data_root, 'surreal_train.h5')
        elif split == 'val':
            data_root = os.path.join(data_root, 'surreal_val.h5')
        else:
            raise NotImplementedError(f'Split {split} is not implemented.')

        self.data_root = data_root
        self.split = split
        self._dataset = None

        # hard-coded for now!
        """
        idxs = []
        for i, im in enumerate(annots['ims']):
            if 'Camera_0' in im:
                idxs.append(i)
        """

        dataset = h5py.File(self.data_root, 'r')
        self.N_imgs = dataset['N_imgs'][()]
        self.cam_inds = dataset['img_pose_indices'][:]
        self.ims = self.cam_inds
        self.N_kps = dataset['Th'].shape[0]
        dataset.close()

        self.num_cams = len(self.ims) // self.N_kps
        self.nrays = cfg.N_rand

    def _init_dataset(self):
        self._dataset = dataset = h5py.File(self.data_root, 'r', swmr=True)
        self.cams = {'R': dataset['R'][:], 'T': dataset['T'][:], 'K': dataset['K']}
        self.body_data = {'Rh': dataset['Rh'][:],
                          'Th': dataset['Th'][:],
                          'vertices': dataset['vertices'][:]}

    def get_mask(self, index):
        msk = self._dataset['fg_masks'][index]
        msk = (msk > 0).astype(np.uint8)

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)

        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk

    def prepare_input(self, index):
        # read xyz, normal, color from the ply file
        kp_idx = index % self.N_kps
        xyz = self.body_data['vertices'][kp_idx]
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        Rh = self.body_data['Rh'][kp_idx:kp_idx+1]
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = self.body_data['Th'][kp_idx].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # transformation augmentation
        xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
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

        if self._dataset is None:
            self._init_dataset()
        img = self._dataset['imgs'][index, ..., :3].astype(np.float32) / 255.
        msk = self.get_mask(index)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind])

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 1. if cfg.white_bkgd else 0.# to white
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
        return self.N_imgs
