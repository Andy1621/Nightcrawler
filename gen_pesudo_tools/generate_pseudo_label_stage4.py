# from winreg import EnumValue
import torch
import pickle
import csv
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gen_label_list_path', type=str, default="./data/total_unlabel.csv")
parser.add_argument('--gen_label_pkl_path_ssv2_best', type=str, default="./exp_pseudo_arid_stage3/uniformer_b32_ssv2_ce/aridx32x224x1x1.pkl")
parser.add_argument('--gen_label_pkl_path_uni_best', type=str, default="./exp_pseudo_arid_stage3/uniformer_b32_k600_ce/aridx32x224x1x1.pkl")
parser.add_argument('--gen_label_pkl_path_mvit_best', type=str, default="./exp_pseudo_arid_stage3/mvit_b32_k600_dp0.3_ce/aridx32x224x1x1.pkl")
parser.add_argument('--gen_label_pkl_path_slowfast_best', type=str, default="./exp_pseudo_arid_stage3/sf32_k600_ce/aridx32x224x1x1.pkl")
parser.add_argument('--path_pseudo_label', type=str, default="./data/pseudo/all_prob_stage4.csv")
args = parser.parse_args()

def get_pseudo_label(idx,ssv2_best_probs,uni_best_probs,mvit_best_probs, slowfast_best_probs):
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


    return new_preds,idx,probs





ssv2_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_ssv2_best, 'rb'))['video_preds']).softmax(-1).numpy()

uni_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_uni_best, 'rb'))['video_preds']).softmax(-1).numpy()

mvit_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_mvit_best, 'rb'))['video_preds']).softmax(-1).numpy()
slowfast_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_slowfast_best, 'rb'))['video_preds']).softmax(-1).numpy()

f=open(args.gen_label_list_path,"r")
idx = f.readlines()
f.close()
idx = [x.strip().split(",")[0] for x in idx]

pred,idx,prob = get_pseudo_label(idx,ssv2_best_probs,uni_best_probs,mvit_best_probs,slowfast_best_probs)

prob = np.array(prob)

pd.DataFrame({"id":idx,"label":pred,"prob0":prob[:,0],"prob1":prob[:,1],"prob2":prob[:,2] \
    ,"prob3":prob[:,3],"prob4":prob[:,4],"prob5":prob[:,5],"prob6":prob[:,6] \
    ,"prob7":prob[:,7],"prob8":prob[:,8],"prob9":prob[:,9],"prob10":prob[:,10]}).to_csv(args.path_pseudo_label,index=False,header=False)
