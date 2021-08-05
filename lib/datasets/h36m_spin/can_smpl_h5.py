import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
import h5py
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData


class Dataset(data.Dataset):
    def __init__(self, data_root, subject, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.subject = subject
        self.split = split
        self._dataset = None

        annots = np.load(ann_file, allow_pickle=True).item()
        # hard-coded for now!
        dataset = h5py.File(os.path.join(self.data_root, f'{subject}_NB_h5py.h5'), 'r', swmr=True)
        if split == 'train_val':
            idxs = []

            val_sets = ['Posing-', 'Walking-', 'Greeting-']
            for i, im_path in enumerate(annots['ims']):
                seq = im_path.split('/')[1]
                is_val = False
                for v in val_sets:
                    if seq.startswith(v):
                        is_val = True

                if not is_val:
                    idxs.append(i)

            idxs = np.array(idxs)
            self.split = 'train'
        else:
            idxs = np.arange(len(annots['ims']))
        self._idx_map = idxs
        self.ims = self._idx_map
        self.cams = annots['cams']

        self.cam_inds = idxs

        self.num_cams = len(self._idx_map)
        self.nrays = cfg.N_rand
        dataset.close()

    def _init_dataset(self):
        self._dataset = h5py.File(os.path.join(self.data_root, f'{self.subject}_NB_h5py.h5'), 'r', swmr=True)
        vertices = h5py.File(os.path.join(self.data_root, f'{self.subject}_NB_vertices.h5'), 'r', swmr=True)
        self.body_data = {'Rh': self._dataset['Rh'][:],
                          'Th': self._dataset['Th'][:],
                          'vertices': self._dataset['vertices'][:]}
        self.bkgds = self._dataset['bkgds'][:]
        self.bkgd_idxs = self._dataset['bkgd_idxs'][:]

    def get_mask(self, index):

        msk = self._dataset['masks'][self.ims[index], ...]#imageio.imread(msk_path)#[..., 0]
        msk = (msk > 0).astype(np.uint8)

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)

        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk

    def prepare_input(self, index):
        # read xyz, normal, color from the ply file
        #vertices_path = os.path.join(self.data_root,
        #                             self.ims[index].replace("imageSequence", "vertices")[:-4] + ".npy")
        #xyz = np.load(vertices_path).astype(np.float32)
        xyz = self.body_data['vertices'][self.ims[index]]
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
        Rh = self.body_data['Rh'][self.ims[index]]
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = self.body_data['Th'][self.ims[index]].astype(np.float32)
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
        #img_path = os.path.join(self.data_root, self.ims[index])
        #img = imageio.imread(img_path).astype(np.float32)[..., :3] / 255.
        img = self._dataset['imgs'][self.ims[index], ..., :3].astype(np.float32) / 255.
        msk = self.get_mask(index)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind])

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        bkgd = self.bkgds[self.bkgd_idxs[self.ims[index]]].copy().astype(np.float32) / 255.
        if cfg.mask_bkgd:
            img[msk == 0] = bkgd[msk == 0]

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
            'bkgd_val': bkgd[coord_[:, 0], coord_[:, 1]],
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