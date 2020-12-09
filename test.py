import argparse
import datetime
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.concordance_correlation_coefficient import concordance_correlation_coefficient as ccc
from data.dataset import Patient
from model.network_model import eTofts, CNNd
from utils.config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--model_path', required=True, type=str)
args = parser.parse_args()

default_config = get_config()
test_config = default_config.test
test_config.update(args.__dict__)

gpu = torch.device('cuda:{}'.format(test_config.gpu))
torch.cuda.set_device(gpu)
torch.multiprocessing.set_sharing_strategy('file_system')
batch = test_config.batch

str_time = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
model_path = test_config.model_path

eTofts_m = eTofts().cuda()
fast_m = CNNd().cuda()
test_path = test_config.dataset
def test(dataset, fast_m, name):
    res_x = []
    res_y = []

    k_trans_x = []
    k_trans_y = []

    vp_x = []
    vp_y = []

    ve_x = []
    ve_y = []

    # print('test.......')
    with torch.no_grad():
        data_load = DataLoader(dataset=dataset, batch_size=batch, shuffle=False, num_workers=4)
        for count, data in enumerate(data_load):
            t10, fited, data, paramters = data
            # fited = fited.cuda()
            t10 = t10.cuda()
            data = data.cuda()

            pre = fast_m(data, t10)

            ve_pos = torch.gt(paramters[:, 0], 0.01)
            out_fit = eTofts_m(pre, t10, data[:, 1, ...])
            pre[:, 0] = pre[:, 0] * 0.2
            pre[:, 1] = pre[:, 1] * 0.1
            pre[:, 2] = pre[:, 2] * 0.6
            k_trans_x.append(paramters[:, 0].cpu().numpy())
            k_trans_y.append(pre[:, 0].cpu().numpy())

            vp_x.append(paramters[:, 1].cpu().numpy())
            vp_y.append(pre[:, 1].cpu().numpy())

            ve_x.append(paramters[ve_pos, 2].cpu().numpy())
            ve_y.append(pre[ve_pos, 2].cpu().numpy())

            res_y.append(np.mean(np.square(out_fit.cpu().numpy() - data[:, 0, ...].cpu().numpy()), axis=1))
            res_x.append(np.mean(np.square(fited.numpy() - data[:, 0, ...].cpu().numpy()), axis=1))

            print('test batch:{batch}/{total_batch}'.format(batch=count,
                                                            total_batch=int((dataset.__len__() - 1) / batch + 1)))

    k_trans_x, k_trans_y = np.concatenate(k_trans_x, axis=0), np.concatenate(k_trans_y, axis=0)
    vp_x, vp_y = np.concatenate(vp_x, axis=0), np.concatenate(vp_y, axis=0)
    ve_x, ve_y = np.concatenate(ve_x, axis=0), np.concatenate(ve_y, axis=0)
    res_x, res_y = np.concatenate(res_x, axis=0), np.concatenate(res_y, axis=0)

    k_trans_mae = np.mean(np.abs(k_trans_x - k_trans_y))
    vp_mae = np.mean(np.abs(vp_x - vp_y))
    ve_mae = np.mean(np.abs(ve_x - ve_y))

    k_trans_ccc = ccc(k_trans_x, k_trans_y)
    vp_ccc = ccc(vp_x, vp_y)
    ve_ccc = ccc(ve_x, ve_y)

    k_trans_nrmse = np.sqrt(np.mean(np.square(k_trans_x - k_trans_y))) / 0.2
    vp_nrmse = np.sqrt(np.mean(np.square(vp_x - vp_y))) / 0.1
    ve_nrmse = np.sqrt(np.mean(np.square(ve_x - ve_y))) / 0.6

    res_x = np.mean(res_x)
    res_y = np.mean(res_y)

    loss_info = 'k_trans,{},{},{},vb,{},{},{},ve,{},{},{},resx,{},resy,{}'.format(
        k_trans_mae, k_trans_ccc, k_trans_nrmse,
        vp_mae, vp_ccc, vp_nrmse,
        ve_mae, ve_ccc, ve_nrmse,
        res_x, res_y
    )
    with open(os.path.join(model_path, 'test_result_{}.txt'.format(name)), 'w') as f:
        f.write(loss_info)
    return loss_info


if not os.path.exists(model_path):
    raise FileNotFoundError(f'{model_path} not exists!')
loss_func = nn.MSELoss(reduction='none')


test_time = str(datetime.datetime.now())

if os.path.isdir(model_path):
    param_path = os.path.join(model_path, 'best_patient.tar')
else:
    param_path = model_path
best_model = torch.load(param_path, map_location=gpu)
fast_m.load_state_dict(best_model)

test_result = []
for pat_dir in os.listdir(test_path):
    test_pat_data = Patient(os.path.join(test_path, pat_dir))
    pat_result = test(test_pat_data, fast_m, pat_dir)
    test_result.append(pat_result)

print('------------Test Result------------')
for pat_name, pat_result in zip(os.listdir(test_path) ,test_result):
    print(pat_result)

print('Test start at', test_time)
print('End at', str(datetime.datetime.now()), f'result path:{model_path}')
