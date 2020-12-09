from torch.utils.data import Dataset, ConcatDataset, random_split
import os
import numpy as np
import random
from utils.config import get_config
config = get_config()
r1 = config.protocol.r1
TR = config.protocol.TR
alpha = config.protocol.alpha/180*np.pi
deltt = config.protocol.deltt/60
data_root = config.train.dataset
from model.eTofts import eTofts_model
class Patient(Dataset):
    def __init__(self, root):
        self.root = root
        self.data = []
        self.cp = self.read_cp()
        self.read_data()

    def __getitem__(self, item):
        data = self.data[item]
        T10 = np.array([data.get('T10'),])
        raw_data = data.get('dce_data')
        fited = data.get('fited_data')
        cp = self.cp
        s0 = np.mean(raw_data[:6])/self.s(T10)
        raw_data = raw_data/s0*20
        fited = fited/s0*20
        param = data.get('param')
        # keys = {'cp', 'T1b0', 'TR', 'TE', 'r1', 'h', 'fw', 'flip_angle'}
        data = np.concatenate((raw_data[np.newaxis, ...], cp[np.newaxis, ...]), axis=0)
        return T10.astype(np.float32), fited.astype(np.float32), data.astype(np.float32), param.astype(np.float32)
    def __len__(self):
        return len(self.data)

    def s(self, T10):
        R1 = 1 / T10
        s = (1 - np.exp(-TR * R1)) * np.sin(alpha) / (1 - np.exp(-TR * R1) * np.cos(alpha))
        return s
    def read_cp(self):
        info_file = os.path.join(self.root, 'cp.npy')
        cp = np.load(info_file, allow_pickle=True)
        cpm = np.argmax(cp)
        return cp[cpm-12:cpm + 100]
    def read_data(self):
        eTofts_files = os.listdir(self.root)
        eTofts_files.remove('cp.npy')
        eTofts_files.sort(key=lambda x: int((os.path.splitext(x)[0]).split('_')[1]))
        for file in eTofts_files:
            # if 'tumor' not in file:
            #     continue
            data = np.load(os.path.join(self.root, file), allow_pickle=True)
            self.data.extend(data)
    def snr(self, item):
        t10, fited, data, par = self.__getitem__(item)
        noise = data[0, :6] - fited[:6]
        signal_power = np.sum(np.square(fited[:6]))
        noise_power = np.sum(np.square(noise[:6]))
        level = np.sqrt(noise_power/signal_power)
        snr = 10*np.log10(signal_power/noise_power)
        return level, snr
class SimulatePatient(Dataset):
    def __init__(self, root, length=100000):
        self.root = root
        self.cps = self.read_cps()
        self.len = length
    def __getitem__(self, item):
        np.random.seed()
        T10 = self.generate_t1()
        ktrans = self.generate_ktrans()
        vp = self.generate_vp()
        ve = self.generate_ve()
        cp = self.generate_cp()
        s0 = 20
        fited = eTofts_model(ktrans, vp, ve, T10, cp) * s0
        raw_data = add_gaussian_noise(fited, 0.01)
        # keys = {'cp', 'T1b0', 'TR', 'TE', 'r1', 'h', 'fw', 'flip_angle'}
        data = np.concatenate((raw_data[np.newaxis, ...], cp[np.newaxis, ...]), axis=0)
        param = np.array([ktrans, vp, ve])
        return np.array([T10, ]).astype(np.float32), fited.astype(np.float32), data.astype(np.float32), param.astype(np.float32)
    def __len__(self):
        return self.len

    def s(self, T10):
        R1 = 1 / T10
        s = (1 - np.exp(-TR * R1)) * np.sin(alpha) / (1 - np.exp(-TR * R1) * np.cos(alpha))
        return s
    def read_cps(self):
        cps = []
        for pats in os.listdir(self.root):
            if pats in ['patient23', 'patient59', 'patient85']:
                continue
            info_file = os.path.join(self.root, pats, 'cp.npy')
            cp = np.load(info_file, allow_pickle=True)
            cpm = np.argmax(cp)
            cps.append(cp[cpm-12:cpm + 100])
        return cps
    def generate_cp(self):
        cp1id, cp2id = np.random.choice(len(self.cps), 2)
        cp1, cp2 = self.cps[cp1id], self.cps[cp2id]
        lam = np.random.random()
        cp = lam*cp1 + (1-lam)*cp2
        return cp
    def generate_ktrans(self):
        log_or_line = np.random.random() >= 0.5
        lam = np.random.random()
        if log_or_line:  # line
            return lam*0.00001 + (1-lam) * 0.2
        else:
            log_trans = lam*(-5) + (1-lam) * np.log10(0.2)
            return np.power(10, log_trans)
    def generate_vp(self):
        lam = np.random.random()
        return lam*0.0005 + (1-lam) * 0.1
    def generate_ve(self):
        lam = np.random.random()
        return lam * 0.04 + (1 - lam) * 0.6
    def generate_t1(self):
        lam = np.random.random()
        return lam * 0.8 + (1 - lam) * 3.5
    def snr(self, item):
        t10, fited, data, par = self.__getitem__(item)
        noise = fited[:6] - data[0, :6]
        signal_power = np.sum(np.square(fited[:6]))
        noise_power = np.sum(np.square(noise[:6]))
        level = np.sqrt(noise_power/signal_power)
        snr = 10*np.log10(signal_power/noise_power)
        return level, snr
    def snr_full(self, item):
        t10, fited, data, par = self.__getitem__(item)
        noise = fited - data[0, :]
        signal_power = np.sum(np.square(fited))
        noise_power = np.sum(np.square(noise))
        level = np.sqrt(noise_power/signal_power)
        snr = 10*np.log10(signal_power/noise_power)
        return level, snr

class MixDataset(Dataset):
    def __init__(self, pat_data):
        self.patient = pat_data
        self.simulate = SimulatePatient(data_root, 100000)
        self.patient_length = self.patient.__len__()
    def __getitem__(self, item):
        if random.random() < 0.5:
            return self.patient.__getitem__(int(random.random()*self.patient_length))
        else:
            return self.simulate.__getitem__(int(random.random()*100000))
    def __len__(self):
        return 200000
def add_gaussian_noise(signal, noise_per):
    rms = np.sqrt(np.mean(np.square(signal[:6])))
    n_std = rms*noise_per
    noise = np.random.randn(*signal.shape)*n_std
    return signal+noise
def patients_dataset():
    patients = []
    for pats in os.listdir(data_root):
        if pats in ['patient23', 'patient59', 'patient85']:
            continue
        patients.append(Patient(os.path.join(data_root, pats)))
    sets = ConcatDataset(patients)
    train_len = int(sets.__len__()*0.75)
    train, valid = random_split(sets, [train_len, sets.__len__()-train_len])
    return train, valid
def mix_dataset():
    pat_train, pat_valid = patients_dataset()
    train, valid = MixDataset(pat_train), MixDataset(pat_valid)
    return train, valid
def synthetic_dataset():

    train = SimulatePatient(data_root, 100000)
    valid = SimulatePatient(data_root, 50000)
    return train, valid
