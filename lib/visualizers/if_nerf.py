import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import imageio
import os


class Visualizer:
    def visualize(self, output, batch):
        rgb_pred = output['rgb_map'].view(1, -1, 3)[0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        print('mse: {}'.format(np.mean((rgb_pred - rgb_gt) ** 2)))

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        """
        if cf.white_bkgd:
            img_pred = np.ones((H, W, 3))
        else:
            img_pred = np.zeros((H, W, 3))
        """
        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred

        if not 'img_gt' in batch:
            img_gt = np.zeros((H, W, 3))
            img_gt[mask_at_box] = rgb_gt
        else:
            img_gt = batch['img_gt'][0].cpu().numpy()

        '''
        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img_pred)
        ax2.imshow(img_gt)
        plt.show()
        '''
        frame_root = 'data/result/if_nerf/{}/'.format(cfg.exp_name)
        index = batch['index']

        pred_path = os.path.join(frame_root, 'pred')
        gt_path = os.path.join(frame_root, 'gt')
        os.system('mkdir -p {}'.format(pred_path))
        os.system('mkdir -p {}'.format(gt_path))
        imageio.imwrite(os.path.join(pred_path, "%d.png" % index), (img_pred * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(gt_path, "%d.png" % index), (img_gt * 255).astype(np.uint8))

