import os

from lib.config import cfg


class DatasetCatalog(object):
    human = cfg.human
    subject = cfg.subject
    dataset_attrs = {
        'Human{}_0001_Train'.format(human): {
            'data_root': 'data/zju_mocap/CoreView_{}'.format(human),
            'human': 'CoreView_{}'.format(human),
            'ann_file':
            'data/zju_mocap/CoreView_{}/annots.npy'.format(human),
            'split': 'train'
        },
        'Human{}_0001_Test'.format(human): {
            'data_root': 'data/zju_mocap/CoreView_{}'.format(human),
            'human': 'CoreView_{}'.format(human),
            'ann_file':
            'data/zju_mocap/CoreView_{}/annots.npy'.format(human),
            'split': 'test'
        },
        'Human362_0001_Train': {
            'data_root': 'data/zju_mocap/CoreView_362',
            'human': 'CoreView_362',
            'ann_file': 'data/zju_mocap/CoreView_362/annots.npy',
            'split': 'train'
        },
        'Human362_0001_Test': {
            'data_root': 'data/zju_mocap/CoreView_362',
            'human': 'CoreView_362',
            'ann_file': 'data/zju_mocap/CoreView_362/annots.npy',
            'split': 'test'
        },
        'Human326_0001_Train': {
            'data_root': 'data/zju_mocap/CoreView_326',
            'human': 'CoreView_326',
            'ann_file': 'data/zju_mocap/CoreView_326/annots.npy',
            'split': 'train'
        },
        'Human326_0001_Test': {
            'data_root': 'data/zju_mocap/CoreView_326',
            'human': 'CoreView_326',
            'ann_file': 'data/zju_mocap/CoreView_326/annots.npy',
            'split': 'test'
        },
        'Human302_0001_Train': {
            'data_root': 'data/zju_mocap/CoreView_302',
            'human': 'CoreView_302',
            'ann_file': 'data/zju_mocap/CoreView_302/annots.npy',
            'split': 'train'
        },
        'Human302_0001_Test': {
            'data_root': 'data/zju_mocap/CoreView_302',
            'human': 'CoreView_302',
            'ann_file': 'data/zju_mocap/CoreView_302/annots.npy',
            'split': 'test'
        },
        'Human329_0001_Train': {
            'data_root': 'data/zju_mocap/CoreView_329',
            'human': 'CoreView_329',
            'ann_file': 'data/zju_mocap/CoreView_329/annots.npy',
            'split': 'train'
        },
        'Human329_0001_Test': {
            'data_root': 'data/zju_mocap/CoreView_329',
            'human': 'CoreView_329',
            'ann_file': 'data/zju_mocap/CoreView_329/annots.npy',
            'split': 'test'
        },
        'Female_1_casual_Train': {
            'data_root': 'data/people_snapshot/female-1-casual',
            'split': 'train'
        },
        'Female_1_casual_Test': {
            'data_root': 'data/people_snapshot/female-1-casual',
            'split': 'test'
        },
        'Female_3_casual_Train': {
            'data_root': 'data/people_snapshot/female-3-casual',
            'split': 'train'
        },
        'Female_3_casual_Test': {
            'data_root': 'data/people_snapshot/female-3-casual',
            'split': 'test'
        },
        'Male_2_casual_Train': {
            'data_root': 'data/people_snapshot/male-2-casual',
            'split': 'train'
        },
        'Male_2_casual_Test': {
            'data_root': 'data/people_snapshot/male-2-casual',
            'split': 'test'
        },
        'Female_4_casual_Train': {
            'data_root': 'data/people_snapshot/female-4-casual',
            'split': 'train'
        },
        'Female_4_casual_Test': {
            'data_root': 'data/people_snapshot/female-4-casual',
            'split': 'test'
        },
        'Male_3_casual_Train': {
            'data_root': 'data/people_snapshot/male-3-casual',
            'split': 'train'
        },
        'Male_3_casual_Test': {
            'data_root': 'data/people_snapshot/male-3-casual',
            'split': 'test'
        },
        'Male_5_outdoor_Train': {
            'data_root': 'data/people_snapshot/male-5-outdoor',
            'split': 'train'
        },
        'Male_5_outdoor_Test': {
            'data_root': 'data/people_snapshot/male-5-outdoor',
            'split': 'test'
        },
        'Male_2_outdoor_Train': {
            'data_root': 'data/people_snapshot/male-2-outdoor',
            'split': 'train'
        },
        'Male_2_outdoor_Test': {
            'data_root': 'data/people_snapshot/male-2-outdoor',
            'split': 'test'
        },
        'Female_8_plaza_Train': {
            'data_root': 'data/people_snapshot/female-8-plaza',
            'split': 'train'
        },
        'Female_8_plaza_Test': {
            'data_root': 'data/people_snapshot/female-8-plaza',
            'split': 'test'
        },
        'Female_6_plaza_Train': {
            'data_root': 'data/people_snapshot/female-6-plaza',
            'split': 'train'
        },
        'Female_6_plaza_Test': {
            'data_root': 'data/people_snapshot/female-6-plaza',
            'split': 'test'
        },
        'Female_7_plaza_Train': {
            'data_root': 'data/people_snapshot/female-7-plaza',
            'split': 'train'
        },
        'Female_7_plaza_Test': {
            'data_root': 'data/people_snapshot/female-7-plaza',
            'split': 'test'
        },
        'H36M_S9P_Train': {
            'data_root': 'data/h36m/S9/Posing',
            'split': 'train'
        },
        'H36M_S9P_Test': {
            'data_root': 'data/h36m/S9/Posing',
            'split': 'test'
        },
        'H36M_S11G_Train': {
            'data_root': 'data/h36m/S11/Greeting',
            'split': 'train'
        },
        'H36M_S11G_Test': {
            'data_root': 'data/h36m/S11/Greeting',
            'split': 'test'
        },
        # MIXAMO
        'Mixamo_{}_Train'.format(subject): {
            'data_root': 'data/mixamo/',
            'subject': '{}'.format(subject),
            'ann_file':
            'data/mixamo/{}/annots.npy'.format(subject),
            'split': 'train'
        },
        'Mixamo_{}_Test'.format(subject): {
            'data_root': 'data/mixamo/',
            'subject': '{}'.format(subject),
            'ann_file':
            'data/mixamo/{}/annots.npy'.format(subject),
            'split': 'test'
        },
        # PerfCap
        'Perfcap_{}_Train'.format(subject): {
            'data_root': 'data/',
            'subject': '{}'.format(subject),
            'ann_file': 'data/MonoPerfCap/{}/annots.npy'.format(subject),
            'split': 'train',
        },
        'Perfcap_{}_Test'.format(subject): {
            'data_root': 'data/',
            'subject': '{}'.format(subject),
            'ann_file': 'data/MonoPerfCap/{}/annots.npy'.format(subject),
            'split': 'test',
        },
        # H36M - SPIN
        'H36M_SPIN_{}_Train'.format(subject): {
            'data_root': 'data/h36m',
            'subject': '{}'.format(subject),
            'ann_file': 'data/h36m/{}/annots.npy'.format(subject),
            'split': 'train',
        },
        'H36M_SPIN_{}_Test'.format(subject): {
            'data_root': 'data/h36m',
            'subject': '{}'.format(subject),
            'ann_file': 'data/h36m/{}/annots.npy'.format(subject),
            'split': 'test',
        },
        'RENDER_SPIN_perfcap_Test': {
            'data_root': 'perfcap',
            'subject': '{}'.format(subject),
            'split': '{}'.format(cfg.render_type),
        },
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
