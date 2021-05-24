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
s9_idx = [121, 1000, 1059, 1600, 1815, 3014, 3702, 4980]
h36m_s9 = {
    'data_h5': 'data/h36m/S9_processed_deeplab_crop3.h5',
    'retarget': set_dict(s9_idx, length=5),
    'bullet': set_dict(s9_idx, undo_rot=True,
                       center_cam=True),
    'interpolate': set_dict(s9_idx, n_step=10, undo_rot=True,
                            center_cam=True),
}

s11_idx = [213, 656, 904, 1559, 1815, 2200, 2611, 2700]
h36m_s11 = {
    'data_h5': 'data/h36m/S11_processed_deeplab_crop3.h5',
    'retarget': set_dict(s11_idx, length=5),
    'bullet': set_dict(s11_idx, undo_rot=True,
                       center_cam=True),
    'interpolate': set_dict(s11_idx, n_step=10, undo_rot=True,
                            center_cam=True),
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
              600, 630, 660, 720, 760, 810, 1030, 1080, 1130]
perfcap_weipeng = {
    'data_root': 'data/',
    'data_h5': 'data/MonoPerfCap/Weipeng_outdoor/Weipeng_outdoor_processed.h5',
    'ann_file': 'data/MonoPerfCap/Weipeng_outdoor/annots.npy',
    'retarget': set_dict(weipeng_idx, length=30, skip=1),
    'bullet': set_dict(weipeng_idx),
    'interpolate': set_dict(weipeng_idx, n_step=10, undo_rot=True,
                            center_cam=True),
    'bubble': set_dict(weipeng_idx, n_step=30),
}

nadia_idx = [0, 65, 100, 125, 230, 280, 410, 560, 600, 630, 730, 770,
             830, 910, 1010, 1040, 1070, 1100, 1285, 1370]
perfcap_nadia = {
    'data_root': 'data/',
    'data_h5': 'data/MonoPerfCap/Nadia_outdoor/Nadia_outdoor_processed.h5',
    'ann_file': 'data/MonoPerfCap/Nadia_outdoor/annots.npy',
    'retarget': set_dict(nadia_idx, length=30, skip=1),
    'bullet': set_dict(nadia_idx),
    'interpolate': set_dict(nadia_idx, n_step=10, undo_rot=True,
                            center_cam=True),
    'bubble': set_dict(nadia_idx, n_step=30),
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


