import argparse
import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.concordance_correlation_coefficient import concordance_correlation_coefficient as ccc
from data.dataset import patients_dataset, synthetic_dataset, Patient, mix_dataset
from model.network_model import eTofts, CNNd
from utils.config import get_config
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--weight', default=40, type=int)
parser.add_argument('--stop_num', default=5, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--name', default='train', type=str)
parser.add_argument('--strategy', default='mix', choices=['synthetic', 'patient', 'fine_tune', 'mix'], type=str)
args = parser.parse_args()
# merge config
default_config = get_config()
train_config = default_config.train
train_config.update(args.__dict__)
# apply config
gpu = torch.device('cuda:{}'.format(train_config.gpu))
torch.cuda.set_device(gpu)
torch.multiprocessing.set_sharing_strategy('file_system')
stop_num = train_config.stop_num
lr = train_config.lr
fit_weight = train_config.weight
strategy = train_config.strategy
max_epoch = train_config.max_epoch
batch = train_config.batch

str_time = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
model_path = os.path.join('exp/{}'.format(train_config.name),
                          strategy + '_weight{0}_stop{1}_lr{2}_{3}'.format(fit_weight, stop_num, lr, str_time))
if strategy == 'mix':
    train_set_simu, valid_set_simu = mix_dataset()
    train_set_pat, valid_set_pat = train_set_simu, valid_set_simu
elif strategy == 'fine-tune':
    train_set_simu, valid_set_simu = synthetic_dataset()
    train_set_pat, valid_set_pat = patients_dataset()
elif strategy == 'patient':
    train_set_simu, valid_set_simu = patients_dataset()
    train_set_pat, valid_set_pat = train_set_simu, valid_set_simu
elif strategy == 'synthetic':
    train_set_simu, valid_set_simu = synthetic_dataset()
    train_set_pat, valid_set_pat = train_set_simu, valid_set_simu
else:
    raise ValueError('No valid training strategy specified')

eTofts_m = eTofts().cuda()
fast_m = CNNd().cuda()


def train(train_set, valid_set, fast_m, optimizer, name='simulate'):
    '''
    :param train_set:
    :param valid_set:
    :param fast_m:
    :param optimizer:
    :param name:
    :return:
    '''
    min_loss = torch.FloatTensor([float('inf'), ])
    global stop_num
    ealy_stop = stop_num
    global fit_weight
    print('fit weight', fit_weight)
    for epoch in range(max_epoch):
        train_l, valid_l = 0, 0
        c_train, c_valid = 0, 0
        dataloader = DataLoader(dataset=train_set, batch_size=batch, shuffle=True, num_workers=2)
        for count, data in enumerate(dataloader):
            t10, fited, data, paramters = data
            # fited = fited.cuda()
            t10 = t10.cuda()
            data = data.cuda()
            paramters[:, 0] = paramters[:, 0] * 5
            paramters[:, 1] = paramters[:, 1] * 10
            paramters[:, 2] = paramters[:, 2] * 5 / 3
            paramters = paramters.cuda()

            pre = fast_m(data, t10)
            out_fit = eTofts_m(pre, t10, data[:, 1, ...])

            pre_loss = loss_func(pre, paramters)

            loss1 = torch.mean(pre_loss)
            loss2 = torch.mean(loss_func(out_fit, data[:, 0, :])) * fit_weight
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            print(
                'train epoch:{epoch},batch:{batch}/{total_batch},param loss:{loss1:.8f},fit loss:{loss2:.8f},loss:{loss:.8f},--stop:{stop}'.format(
                    epoch=epoch, batch=count, total_batch=int((train_set.__len__() - 1) / batch + 1),
                    loss1=loss1.item(), loss2=loss2.item(), loss=loss.item(), stop=ealy_stop))
            if count == 3:
                break
        with torch.no_grad():
            # k_loss, pb_loss, po_loss = torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0])
            valid_load = DataLoader(dataset=valid_set, batch_size=batch, shuffle=False, num_workers=2)
            for count, data in enumerate(valid_load):
                t10, fited, data, paramters = data
                # fited = fited.cuda()
                t10 = t10.cuda()
                data = data.cuda()
                paramters[:, 0] = paramters[:, 0] * 5
                paramters[:, 1] = paramters[:, 1] * 10
                paramters[:, 2] = paramters[:, 2] * 5 / 3
                paramters = paramters.cuda()

                pre = fast_m(data, t10)
                out_fit = eTofts_m(pre, t10, data[:, 1, ...])

                pre_loss = loss_func(pre, paramters)

                loss1 = torch.mean(pre_loss)
                loss2 = torch.mean(loss_func(out_fit, data[:, 0, :])) * fit_weight
                v_loss = loss1 + loss2

                valid_l = valid_l + v_loss*paramters.size(0)
                c_valid = c_valid + paramters.size(0)
                if count % 10 == 0:
                    c = 3
                    r = data[c, 0, :].cpu().detach().numpy()
                    p = out_fit[c, :].cpu().detach().numpy()
                    f = fited[c, :].numpy()
                    plt.plot(r, label='target')
                    plt.plot(p, label='predict')
                    plt.plot(f, label='fit')
                    # plt.plot(t, '*',label='test')
                    plt.legend()
                    plt.title(os.path.split(model_path)[1] + 'valid')
                    plt.pause(0.001)
                    plt.cla()
                print('valid epoch:{epoch},batch:{batch}/{v_batch},loss:{loss},--stop:{stop}'.format(
                    epoch=epoch, batch=count, v_batch=int((valid_set.__len__() - 1) / batch + 1),
                    loss=valid_l.item() / c_valid, stop=ealy_stop))
                if count == 3:
                    break
        if min_loss > valid_l.cpu():
            torch.save(fast_m.state_dict(), os.path.join(model_path, 'best_{}.tar'.format(name)))
            min_loss = valid_l.cpu()
            ealy_stop = stop_num
        else:
            ealy_stop = ealy_stop - 1
        print('*** valid loss', valid_l.item() / c_valid, 'min loss:', min_loss.item() / c_valid, 'ealy_stop',
              ealy_stop)
        if ealy_stop and epoch < max_epoch - 1:
            continue
        return epoch


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
    with open(os.path.join(model_path, 'train_result_{}.txt'.format(name)), 'w') as f:
        f.write(loss_info)


if not os.path.exists(model_path):
    os.makedirs(model_path)
loss_func = nn.MSELoss(reduction='none')

optimizer = Adam(fast_m.parameters(), lr=lr)
if not os.path.exists(model_path):
    os.makedirs(model_path)

train_time = str(datetime.datetime.now())
s_epoch = train(train_set_simu, valid_set_simu, fast_m, optimizer, name='simulate')
best_model = torch.load(os.path.join(model_path, 'best_simulate.tar'), map_location=gpu)
fast_m.load_state_dict(best_model)

optimizer = Adam(fast_m.parameters(), lr=lr / 10)
tune_time = str(datetime.datetime.now())
t_epoch = train(train_set_pat, valid_set_pat, fast_m, optimizer, name='patient')

best_model = torch.load(os.path.join(model_path, 'best_patient.tar'), map_location=gpu)
fast_m.load_state_dict(best_model)

test(Patient('/ext2/fk/data/fast_eTofts_data/patient23'), fast_m, 'patient23')
# test(Patient('/ext2/fk/data/fast_eTofts_data/patient59'), fast_m, 'patient59')
# test(Patient('/ext2/fk/data/fast_eTofts_data/patient85'), fast_m, 'patient85')

print('First train start at', train_time, f'total {s_epoch} epoch')
print('Second train start at', tune_time, f'total {t_epoch} epoch')
print('End at', str(datetime.datetime.now()), f'result path:{model_path}')
