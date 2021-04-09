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
from lib.utils import render_utils


class Dataset(data.Dataset):
    def __init__(self, data_root, subject, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.subject = subject
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        # hard-coded for now!
        self.cams = annots['cams']
        num_cams = 140 * 4
        idxs = np.arange(len(annots['ims']))[-num_cams:4]
        self.cams = annots['cams']

        self.ims = np.array(annots['ims'])[idxs]
        self.train_ims = self.ims.reshape(-1, 1)
        self.cam_inds = idxs

        """
        num_cams = len(self.cams['K'])
        test_view = [i for i in range(num_cams) if i not in cfg.training_view]
        view = cfg.training_view if split == 'train' else test_view

        i = 0
        i = i + cfg.begin_i
        i_intv = cfg.i_intv
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + cfg.ni * i_intv][::i_intv]
        ]).ravel()
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + cfg.ni * i_intv][::i_intv]
        ]).ravel()
        self.num_cams = len(view)
        """

        K, RT = self.load_cam(ann_file)
        self.Ks = np.array(K)[idxs].astype(np.float32)
        self.RT = np.array(RT)[idxs].astype(np.float32)
        import pdb; pdb.set_trace()
        """
        self.train_ims = np.array([
            np.array(ims_data['ims'])[cfg.training_view]
            for ims_data in annots['ims'][i:i + cfg.ni * i_intv][::i_intv]
        ])
        """

        self.num_cams = len(self.ims)
        self.nrays = cfg.N_rand

    def load_cam(self):
        cams = self.cams

        K = []
        RT = []
        lower_row = np.array([[0., 0., 0., 1.]])

        for i in range(len(cams['K'])):
            K.append(np.array(cams['K'][i]))
            K[i][:2] = K[i][:2] * cfg.ratio

            r = np.array(cams['R'][i])
            t = np.array(cams['T'][i]) / 1000.
            r_t = np.concatenate([r, t], 1)
            RT.append(np.concatenate([r_t, lower_row], 0))

        return K, RT

    def prepare_input(self, index):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root,
                                     self.train_ims[index].replace("ImageSequence", "vertices")[:-4] + ".npy")
        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the origin bounds for point sampling
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
        params_path = os.path.join(self.data_root,
                                   self.train_ims[index].replace("ImageSequence", "smpl")[:-4] + ".npy")
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # move the point cloud to the canonical frame, which eliminates the influence of translation
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

        return feature, coord, out_sh, can_bounds, bounds, Rh, Th

    def get_mask(self, i):
        ims = self.train_ims[i]
        msks = []

        for nv in range(len(ims)):
            im = ims[nv]

            msk_path = os.path.join(self.data_root,
                                    im.replace("ImageSequence", "Masks"))
            msk = imageio.imread(msk_path)[..., :1]
            msk = (msk != 0).astype(np.uint8)

            K = self.Ks[nv].copy()
            K[:2] = K[:2] / cfg.ratio

            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk = cv2.dilate(msk.copy(), kernel)

            msks.append(msk)

        return msks

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims.ravel()[index])
        feature, coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(
            index)

        i = index
        msks = self.get_mask(i)

        # reduce the image resolution by ratio
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        msks = [
            cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            for msk in msks
        ]
        msks = np.array(msks)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        K[:2] = K[:2] * cfg.ratio
        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.
        RT = np.concatenate([R, T], axis=1)

        ray_o, ray_d, near, far, center, scale, mask_at_box = render_utils.image_rays(
            RT, K, can_bounds)

        ret = {
            'feature': feature,
            'coord': coord,
            'out_sh': out_sh,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        i = index #// self.num_cams
        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'i': i,
            'index': index,
            'cam_ind': cam_ind
        }
        ret.update(meta)

        meta = {'msks': msks, 'Ks': self.Ks, 'RT': self.RT}
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
