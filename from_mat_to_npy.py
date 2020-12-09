import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
import argparse
from model.eTofts import fit_eTofts, full_eTofts
parser = argparse.ArgumentParser()
parser.add_argument('--mat', required=True, type=str)
parser.add_argument('--npy', required=True, type=str)
args = parser.parse_args()
pats = [args.mat, ]
save_path = args.npy
if not os.path.exists(save_path):
    os.makedirs(save_path)
def update_data(data,cp,q):
    T10, signal = data['T10'], data['dce_data']
    par = fit_eTofts(T10, cp, signal)
    fited = full_eTofts(*par, T10, cp, signal)

    data.update({'fited_data': fited})
    data.update({'param': par})
    data.update({'dce_data': signal})
    q.put(data)
pool = multiprocessing.Pool()
queue = multiprocessing.Manager().Queue(1024)
for pat in pats:
    pat_num = os.path.split(pat)[1].split('_')[1]
    cp = scio.loadmat(os.path.join(pat, 'Cp.mat'))['Cp'][:, 0]
    cpm = np.argmax(cp)
    save_dir = os.path.join(save_path, 'patient{}'.format(pat_num))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cp = cp[cpm-12:cpm+100]

    np.save(os.path.join(save_dir, 'cp.npy'), cp)
    dces = os.listdir(os.path.join(pat, 'DCE/gaus'))
    # print('pat', pat, dces)
    if 'whole.npy' in dces:
        print('loading', os.path.join(pat, 'DCE/gaus/whole.npy'))
        dce_data = np.load(os.path.join(pat, 'DCE/gaus/whole.npy'))
    else:
        # dces.remove('whole.npy')
        dces.sort(key=lambda x: int((x.split('.')[0])[8:]))
        N = len(dces)
        dce_data = np.zeros((384, 384, 80, N))
        for i in range(N):
            print('loading', i)
            dce = scio.loadmat(os.path.join(pat, 'DCE/gaus', dces[i]))
            dce_data[..., i] = dce['dat']
        np.save(os.path.join(pat, 'DCE/gaus', 'whole.npy'), dce_data)

    T1 = scio.loadmat(os.path.join(pat, 'T10.mat'))['T10']/1000
    # cp = scio.loadmat(os.path.join(os.path.split(os.path.split(pat)[0])[0], 'Cp.mat'))['Cp'][:, 0]
    brain = scio.loadmat(os.path.join(pat, 'mask_whole_brain.mat'))['mask']

    tumor_data = scio.loadmat(os.path.join(pat, 'mask_tumor_segment.mat'))
    tumor_slices = tumor_data['slice'][0]
    tumor_mask = tumor_data['mask']

    p_num = 0
    tumor_datas = []
# def write_tumor():
#     tumor_slice0 = tumor_slices[0] - 1
    error = 0
    for idx, slice_num_mat in enumerate(tumor_slices):
        slice_num = slice_num_mat - 1
        for i in range(384):
            for j in range(384):
                if tumor_mask[i, j, idx] < 1:
                    continue
                if T1[i, j, slice_num] < 0.8 or T1[i, j, slice_num]>3.5:
                    continue
                if brain[i, j, slice_num] < 1:
                    continue
                # print(T1[i, j, slice_num].shape, dce_data[i, j, slice_num, :].shape, cp.shape)
                raw_dce = dce_data[i, j, slice_num, :]
                pads = np.mean(raw_dce[:4])
                dce = raw_dce[cpm-12:cpm+100]
                data = {'T10': T1[i, j, slice_num], 'dce_data': dce, 'position': (i, j, slice_num)}
                x = pool.apply_async(update_data, args=(data, cp, queue))
                if p_num == 0:
                    if x.get() is not None:
                        raise BaseException(str(x.get()))
                p_num = p_num + 1
    while p_num:
        data = queue.get()
        tumor_datas.append(data)
        print(p_num, pat, 'tumor')
        p_num = p_num - 1
    tumor_count = len(tumor_datas)
    np.save(os.path.join(save_dir, 'tumor_{}'.format(tumor_count)), tumor_datas)

    normal_data = scio.loadmat(os.path.join(pat, 'mask_NAWM_segment.mat'))
    normal_slices = normal_data['slice'][0]
    normal_mask = normal_data['mask']
    p_num = 0
    normal_datas = []
    for idx, slice_num_mat in enumerate(normal_slices):
        slice_num = slice_num_mat - 1
        for i in range(384):
            for j in range(384):
                if normal_mask[i, j, idx] < 1:
                    continue
                if T1[i, j, slice_num] < 0.8 or T1[i, j, slice_num] > 3.5:
                    continue
                if brain[i, j, slice_num] < 1:
                    continue
                raw_dce = dce_data[i, j, slice_num, :]
                pads = np.mean(raw_dce[:4])
                dce = raw_dce[cpm - 12:cpm + 100]
                data = {'T10': T1[i, j, slice_num], 'dce_data': dce, 'position': (i, j, slice_num)}
                x = pool.apply_async(update_data, args=(data, cp, queue))
                if p_num == 0:
                    if x.get() is not None:
                        raise BaseException(str(x.get()))
                p_num = p_num + 1
    while p_num:
        data = queue.get()
        normal_datas.append(data)
        print(p_num, pat, 'normal')
        p_num = p_num - 1
    normal_count = len(normal_datas)
    np.save(os.path.join(save_dir, 'normal_{}'.format(normal_count)), normal_datas)




