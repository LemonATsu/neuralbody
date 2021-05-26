import numpy as np


RenderCatalog = {
    'h36m': None,
    'surreal': None,
    'perfcap': None,
    'mixamo': None,
    '3dhp': None,
}

def set_dict(selected_idxs, **kwargs):
    return {'selected_idxs': np.array(selected_idxs), **kwargs}

# H36M
# TODO: currently use the same set of idx for one dataset.
#       can set different index for different things
s9_idx = [121, 500, 1000, 1059, 1300, 1600, 1815, 3014, 3702, 4980]
h36m_s9 = {
    'data_root': 'data/h36m',
    'data_h5': 'data/h36m/S9_processed_deeplab_crop3.h5',
    'ann_file': 'data/h36m/S9/annots.npy',
    'refined': 'neurips21_ckpt/trained/ours/h36m/s9_sub64_500k.tar',
    'retarget': set_dict(s9_idx, length=5),
    'bullet': set_dict(s9_idx, undo_rot=True,
                       center_cam=True),
    'interpolate': set_dict(s9_idx, n_step=10, undo_rot=True,
                            center_cam=True),
    'bubble': set_dict(s9_idx, n_step=30),
}

s11_idx = [213, 656, 904, 1559, 1815, 2200, 2611, 2700, 3110, 3440, 3605]
h36m_s11 = {
    'data_root': 'data/h36m',
    'data_h5': 'data/h36m/S11_processed_deeplab_crop3.h5',
    'ann_file': 'data/h36m/S11/annots.npy',
    'refined': 'neurips21_ckpt/trained/ours/h36m/s11_sub64_500k.tar',
    'retarget': set_dict(s11_idx, length=5),
    'bullet': set_dict(s11_idx, undo_rot=True,
                       center_cam=True),
    'interpolate': set_dict(s11_idx, n_step=10, undo_rot=True,
                            center_cam=True),
    'bubble': set_dict(s11_idx, n_step=30),
}

# SURREAL
easy_idx = [10, 70, 350, 420, 490, 910, 980, 1050]
surreal_easy = {
    'data_h5': 'data/surreal_hr/surreal_hr.h5',
    'retarget': set_dict(easy_idx, length=25, skip=1),
    'bullet': set_dict(easy_idx),
}
hard_idx = [140, 210, 280, 560, 630, 700, 770]
surreal_hard = {
    'data_h5': 'data/surreal_hr/surreal_hr.h5',
    'retarget': set_dict(hard_idx, length=60, skip=5),
    'bullet': set_dict(hard_idx),
}

# PerfCap
weipeng_idx = [0, 50, 100, 150, 200, 250, 300, 350, 430, 480, 560,
               600, 630, 660, 690, 720, 760, 810, 850, 900, 950,
               1030, 1080, 1120]
perfcap_weipeng = {
    'data_root': 'data/',
    'data_h5': 'data/MonoPerfCap/Weipeng_outdoor/Weipeng_outdoor_processed.h5',
    'ann_file': 'data/MonoPerfCap/Weipeng_outdoor/annots.npy',
    'refined': 'neurips21_ckpt/trained/ours/perfcap/weipeng_tv_500k.tar',
    'idx_map': np.arange(1151),
    'retarget': set_dict(weipeng_idx, length=30, skip=2),
    'bullet': set_dict(weipeng_idx),
    'interpolate': set_dict(weipeng_idx, n_step=10, undo_rot=True,
                            center_cam=True),
    'bubble': set_dict(weipeng_idx, n_step=30),
    'all': set_dict(np.arange(1151)),
}

nadia_idx = [0, 65, 100, 125, 230, 280, 410, 560, 600, 630, 730, 770,
             830, 910, 1010, 1040, 1070, 1100, 1285, 1370, 1450, 1495,
             1560, 1595]
perfcap_nadia = {
    'data_root': 'data/',
    'data_h5': 'data/MonoPerfCap/Nadia_outdoor/Nadia_outdoor_processed.h5',
    'ann_file': 'data/MonoPerfCap/Nadia_outdoor/annots.npy',
    'refined': 'neurips21_ckpt/trained/ours/perfcap/nadia_tv_500k.tar',
    'idx_map': np.arange(1635),
    'retarget': set_dict(nadia_idx, length=30, skip=2),
    'bullet': set_dict(nadia_idx),
    'interpolate': set_dict(nadia_idx, n_step=10, undo_rot=True,
                            center_cam=True),
    'bubble': set_dict(nadia_idx, n_step=30),
    'all': set_dict(np.arange(1635)),
}

# Mixamo
james_idx = [20, 78, 138, 118, 1149, 333, 3401, 2221, 4544]
mixamo_james = {
    'data_root': 'data/mixamo',
    'data_h5': 'data/mixamo/James_processed.h5',
    'ann_file': 'data/mixamo/James/annots.npy',
    'idx_map': np.load('data/mixamo/James_selected.npy'),
    'refined': 'neurips21_ckpt/trained/ours/mixamo/james_tv_500k.tar',
    'retarget': set_dict(james_idx, length=30, skip=2),
    'bullet': set_dict(james_idx, n_bullet=10),
    'interpolate': set_dict(james_idx, n_step=10, undo_rot=True,
                            center_cam=True),
    'bubble': set_dict(james_idx, n_step=30),
    'all': set_dict(np.load('data/mixamo/James_selected.npy')),
}

archer_idx = [158, 672, 374, 414, 1886, 2586, 2797, 4147, 4465]
mixamo_archer = {
    'data_root': 'data/mixamo',
    'data_h5': 'data/mixamo/Archer_processed.h5',
    'ann_file': 'data/mixamo/Archer/annots.npy',
    'idx_map': np.load('data/mixamo/Archer_selected.npy'),
    'refined': 'neurips21_ckpt/trained/ours/mixamo/archer_tv_500k.tar',
    'retarget': set_dict(archer_idx, length=30, skip=2),
    'bullet': set_dict(archer_idx, n_bullet=10),
    'interpolate': set_dict(archer_idx, n_step=10, undo_rot=True,
                            center_cam=True),
    'bubble': set_dict(archer_idx, n_step=30),
    'all': set_dict(np.load('data/mixamo/Archer_selected.npy')),
}


RenderCatalog['h36m'] = {
    'S9': h36m_s9,
    'S11': h36m_s11,
}
RenderCatalog['surreal'] = {
    'easy': surreal_easy,
    'hard': surreal_hard,
}
RenderCatalog['perfcap'] = {
    'Weipeng_outdoor': perfcap_weipeng,
    'Nadia_outdoor': perfcap_nadia,
}

RenderCatalog['mixamo'] = {
    'James': mixamo_james,
    'Archer': mixamo_archer,
}


