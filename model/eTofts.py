import numpy as np
from utils.config import get_config
from scipy.optimize import least_squares as ls
from scipy.optimize import lsq_linear
config = get_config()
r1 = config.protocol.r1
TR = config.protocol.TR
alpha = config.protocol.alpha/180*np.pi
deltt = config.protocol.deltt/60
def eTofts_model(ktrans, vp, ve, T10, cp):
    ce = np.zeros_like(cp, dtype=np.float)
    cp_length = np.size(cp)
    R10 = 1/T10
    for t in range(cp_length):
        ce[t] = np.sum(cp[:t+1]*np.exp(ktrans/ve*(np.arange(t+1)-t)*deltt))*deltt
    # print('fit',ce)
    ce = ce * ktrans
    ct = vp*cp + ce
    R1 = R10 + r1*ct
    s = (1-np.exp(-TR*R1))*np.sin(alpha)/(1-np.exp(-TR*R1)*np.cos(alpha))
    return s
def s(T10):
    R1 = 1/T10
    s = (1 - np.exp(-TR * R1)) * np.sin(alpha) / (1 - np.exp(-TR * R1) * np.cos(alpha))
    return s
def target_func(x, T10, cp, signal):
    ktrans, vp, ve = x
    s = eTofts_model(ktrans, vp, ve, T10, cp)
    s0 = np.mean(signal[:6])/np.mean(s[:6])
    return s*s0-signal
def fit_eTofts(T10, cp, signal):
    '''
    Implementation of nonlinear-least-square
    :param T10:
    :param cp:
    :param signal:
    :return:
    '''
    k_trans0 = 0.01
    vb0 = 0.02
    vo0 = 0.2
    x0 = (k_trans0, vb0, vo0)
    bounds = [(1e-5, 0.0005, 0.04), (0.2, 0.1, 0.6)]
    result = ls(target_func, x0, bounds=bounds, args=(T10, cp, signal))
    return result.x
def full_eTofts(ktrans, vp, ve, T10, cp, signal):
    s = eTofts_model(ktrans, vp, ve, T10, cp)
    s0 = np.mean(signal[:6]) / np.mean(s[:6])
    return s * s0
def NLSQ(T10, cp, signal):
    '''
    Implementation of linear least-squares(LLS), "Murase K: Efficient method for calculating kinetic parameters using T1-weighted dynamic contrast-enhanced magnetic resonance imaging. Magnetic Resonance in Medicine 2004; 51:858–862."
    :param T10:
    :param cp:
    :param signal:
    :return:
    '''
    R10 = 1/T10
    s0 = (1 - np.exp(-TR * R10)) * np.sin(alpha) / (1 - np.exp(-TR * R10) * np.cos(alpha))
    M = np.mean(signal[:6])/s0
    R1t = -np.log((signal-M*np.sin(alpha))/(signal*np.cos(alpha)-M*np.sin(alpha)))/TR
    ctis = (R1t - R10)/r1
    cp_intergral = np.zeros_like(cp, dtype=np.float)
    ctis_intergral = np.zeros_like(cp, dtype=np.float)
    cp_length = np.size(cp)
    for t in range(cp_length):
        cp_intergral[t] = np.sum(cp[:t+1])*deltt
    for t in range(cp_length):
        ctis_intergral[t] = np.sum(ctis[:t+1])*deltt
    matrixA = np.concatenate((cp_intergral.reshape((cp_length, 1)), -ctis_intergral.reshape((cp_length, 1)),
                              cp.reshape((cp_length, 1))), axis=1)
    matrixC = ctis
    bounds = [(1e-5, 1e-5, 0.0005), (0.7, 5, 0.1)]
    matrixB = lsq_linear(matrixA, matrixC, bounds=bounds).x
    vp = matrixB[2]
    k2 = matrixB[1]
    ktrans = matrixB[0] - k2*vp
    ve = ktrans/k2
    return np.array([ktrans, vp, ve])

if __name__ == '__main__':
    import os
    import numpy as np
    import time
    root = '/ext/fk/data/fast_eTofts_data'
    old_root = '/ext/fk/data/eTofts_dataset'
    raw_path = os.path.join(root, 'patient23')   #23, 488,488 # 522,71#85, 698, 574
    eTofts_files = os.listdir(raw_path)
    eTofts_files.remove('cp.npy')
    # eTofts_files.sort(key=lambda x: int((os.path.splitext(x)[0]).split('_')[2]))
    p_num = 0
    for file in eTofts_files:
        if 'tumor' not in file:
            continue
        data = np.load(os.path.join(raw_path, file), allow_pickle=True)
        info_file = os.path.join(raw_path, 'cp.npy')
        cp = np.load(info_file, allow_pickle=True)
        cpm = np.argmax(cp)
        cp = cp[cpm - 12:cpm + 100]
        L = len(data)
        c = 0
        lc = 0
        nc = 0
        l_time = 0
        n_time = 0
        for i, d in enumerate(data):
            T10 = np.array([d.get('T10'), ])
            raw_data = d.get('dce_data')
            fited = d.get('fited_data')
            store_param = d.get('param')
            # if store_param[0] < 1e-4:
            #     continue
            store_fit = fited
            tip = time.time()
            new_param = fit_eTofts(T10, cp, raw_data)
            n_time = time.time() - tip + n_time
            # new_fit = full_eTofts(new_param[0], new_param[1], new_param[2], T10, cp, raw_data)
            new_fit2 = full_eTofts2(new_param[0], new_param[1], new_param[2], T10, cp, raw_data)
            tip = time.time()
            nlsq_param = NLSQ(T10, cp, raw_data)
            l_time = time.time() - tip + l_time
            # nlsq_fit = full_eTofts(nlsq_param[0], nlsq_param[1], nlsq_param[2], T10, cp, raw_data)
            nlsq_fit2 = full_eTofts2(nlsq_param[0], nlsq_param[1], nlsq_param[2], T10, cp, raw_data)
            res_n = np.sum((raw_data-new_fit2)**2)
            res_l = np.sum((raw_data - nlsq_fit2) ** 2)
            plt.plot(raw_data, label='raw')
            plt.plot(store_fit, label='store_fit')
            # plt.plot(new_fit, '--', label='new_fit')
            plt.plot(nlsq_fit2, label='nlsq_fit2')
            plt.plot(new_fit2, '--', label='new_fit2')
            # plt.plot(nlsq_fit, label='nlsq_fit')
            plt.legend()

            if res_l > res_n:
                nc = nc + 1
            else:
                lc = lc + 1
            c = c + 1
            print(#np.round(store_param, decimals=6), np.round(new_param, decimals=6), np.round(nlsq_param, decimals=6),
                  i, L, res_l, res_n)
            if i > 1000:
                break
            # plt.show()
            plt.pause(0.001)
            plt.cla()
        print(lc,nc,c,float(lc)/float(c), float(nc)/float(c),'time', l_time, n_time)
        exit(0)