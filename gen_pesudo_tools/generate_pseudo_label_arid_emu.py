# from winreg import EnumValue
import torch
import pickle
import csv
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gen_label_list_path', type=str, default="./data/arid_unlabel.csv")
parser.add_argument('--gen_label_pkl_path_ssv2_best', type=str, default="./exp_adapt_bn/uniformer_b32_ssv2/aridx32x224x1x1.pkl")
parser.add_argument('--gen_label_pkl_path_uni_best', type=str, default="./exp_adapt_bn/uniformer_b32_k600/aridx32x224x1x1.pkl")
parser.add_argument('--gen_label_pkl_path_mvit_best', type=str, default="./exp_adapt_bn/mvit_b32_k600_dp0.3/aridx32x224x1x1.pkl")
parser.add_argument('--gen_label_pkl_path_slowfast_best', type=str, default="./exp_adapt_bn/sf32_k600/aridx32x224x1x1.pkl")

args = parser.parse_args()


def get_pseudo_label(cla,th,idx,ssv2_best_probs,uni_best_probs,mvit_best_probs,slowfast_best_probs):
    ssv2_probs = ssv2_best_probs
    ssv2_preds = ssv2_probs.argmax(1)

    indx_7_6 = np.bitwise_or((ssv2_preds==6),(ssv2_preds==7))
    other_indx = np.bitwise_xor(np.ones(len(ssv2_preds)).astype(np.bool),indx_7_6)

    new_probs = np.zeros(ssv2_probs.shape)
    uni_probs = uni_best_probs

    new_probs[other_indx]=ssv2_probs[other_indx]*0.375+uni_probs[other_indx]*0.375+mvit_best_probs[other_indx]*0.1+slowfast_best_probs[other_indx]*0.1
    new_probs[indx_7_6] = ssv2_probs[indx_7_6]*0.8 + uni_probs[indx_7_6]*0.2
    probs = new_probs
    new_preds = new_probs.argmax(1)
    preds = []
    idx_new = []
    probs_new = []


    for i,_ in enumerate(new_probs):
        if new_probs[i].argmax() == cla and max(new_probs[i]) >= th:
            preds.append(new_preds[i])
            idx_new.append(idx[i])
            probs_new.append(probs[i])

    return preds,idx_new,probs_new





ssv2_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_ssv2_best, 'rb'))['video_preds']).softmax(-1).numpy()[3088:]
uni_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_uni_best, 'rb'))['video_preds']).softmax(-1).numpy()[3088:]
mvit_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_mvit_best, 'rb'))['video_preds']).softmax(-1).numpy()[3088:]
slowfast_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_slowfast_best, 'rb'))['video_preds']).softmax(-1).numpy()[3088:]

f=open(args.gen_label_list_path,"r")
idx = f.readlines()
f.close()
idx = [x.strip().split(",")[0] for x in idx]

real_csv_list = []
with open('./data/real_arid_unlabel.csv')as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        real_csv_list.append(row)

real_label_cnt ={}
for i in range(0,11):
    real_label_cnt[i] = 0
real = {}
for i,row in enumerate(real_csv_list):
    real[row[0]]=row[1]
    real_label_cnt[int(row[1])] += 1

cla = 10
check_l = {}
t_list = np.linspace(0, 1, 1001)
print("real_label_cnt",real_label_cnt[cla])
maxx = 0
for th in t_list:
    pred,idx_x,prob = get_pseudo_label(cla,th,idx,ssv2_best_probs,uni_best_probs,mvit_best_probs,slowfast_best_probs)

    prob = np.array(prob)

    t_cnt = 0

    for i,_ in enumerate(idx_x):
        if int(real[idx_x[i]]) == int(pred[i]):
            t_cnt+=1

    ratio = t_cnt/len(pred)

    f1 = (2*(len(pred)/real_label_cnt[cla])*(t_cnt/len(pred)))/(len(pred)/real_label_cnt[cla])+(t_cnt/len(pred))
    if f1>maxx:
        maxx=f1
        print("th:",th)
        print("s",len(pred))
        print("total number of real_label = ",t_cnt)
        print("acc = ",t_cnt/len(pred))
        print("acc2 = ",len(pred)/real_label_cnt[cla])
        print("F1",f1)

