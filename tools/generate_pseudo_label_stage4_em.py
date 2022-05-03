# from winreg import EnumValue
import torch
import pickle
import csv
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gen_label_list_path', type=str, default="/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/data/total_unlabel.csv")
parser.add_argument('--gen_label_pkl_path_ssv2_best', type=str, default="/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/exp_pseudo_arid_stage3/uniformer_b32_ssv2_ce/real_unlabelx32x224x1x1.pkl")
# parser.add_argument('--gen_label_pkl_path_ssv2_last', type=str, default="/mnt/lustre/shangjingjie1/newexp/uniformer_k400_ssv2_bn/last_epoch/pseudo_train_unlabelx32x224x1x1.pkl")
parser.add_argument('--gen_label_pkl_path_uni_best', type=str, default="/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/exp_pseudo_arid_stage3/uniformer_b32_k600_ce/real_unlabelx32x224x1x1.pkl")
# parser.add_argument('--gen_label_pkl_path_uni_last', type=str, default="/mnt/lustre/shangjingjie1/newexp/uniformer_bn2/last_epoch/pseudo_train_unlabelx32x224x1x1.pkl")
parser.add_argument('--gen_label_pkl_path_mvit_best', type=str, default="/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/exp_pseudo_arid_stage3/mvit_b32_k600_dp0.3_ce/real_unlabelx32x224x1x1.pkl")
parser.add_argument('--gen_label_pkl_path_slowfast_best', type=str, default="/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/exp_pseudo_arid_stage3/sf32_k600_ce/real_unlabelx32x224x1x1.pkl")
parser.add_argument('--path_pseudo_label', type=str, default="/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/data/pseudo/all_prob_stage4.csv")
args = parser.parse_args()

label_min_th_map = {0:0.247,1:0.178,2:0.236,3:0.207,4:0.175,5:0.202,6:0.219,7:0.246 \
                    ,8:0.184,9:0.211,10:0.202}

label_mean_th_map = {0:0.554,1:0.418,2:0.384,3:0.425,4:0.388,5:0.504,6:0.772,7:0.733 \
                    ,8:0.494,9:0.582,10:0.590}

label_mean_th_map = {0:0.502,1:0.512,2:0.467,3:0.555,4:0.498,5:0.515,6:0.746,7:0.754 \
                    ,8:0.619,9:0.574,10:0.621}#4292_0.91_prob
                    
label_mean_th_map = {0:0.509,1:0.385,2:0.416,3:0.445,4:0.385,5:0.448,6:0.684,7:0.761 \
                    ,8:0.465,9:0.678,10:0.324}#p

label_mean_th_map = {0:0.763,1:0.545,2:0.298,3:0.509,4:0.496,5:0.715,6:0.659,7:0.885 \
                    ,8:0.589,9:0.723,10:0.693}#stage2
                    #0:524 1:519 2:186 3:368 4:344 5:439  6:441 7:372
                    #8:545 9:369 10:583

label_mean_th_map = {0:0.275,1:0.679,2:0.273,3:0.354,4:0.426,5:0.762,6:0.256,7:0.445 \
                    ,8:0.443,9:0.712,10:0.755}#stage3
                    #0:333 1:526 2:277 3:470 4:509 5:529  6:453 7:454
                    #8:628 9:407 10:578

label_mean_th_map = {0:0.275,1:0.679,2:0.273,3:0.354,4:0.426,5:0.762,6:0.256,7:0.445 \
                    ,8:0.443,9:0.390,10:0.755}#stage3
                    #0:333 1:526 2:277 3:470 4:509 5:529  6:453 7:454
                    #8:628 9:502 10:578

# label_mean_th_map = {0:0.7,1:0.418,2:0.384,3:0.425,4:0.388,5:0.68,6:0.772,7:0.8 \
#                     ,8:0.6,9:0.66,10:0.590}


label_th_list = []
for i in range(11):
    a = []
    label_th_list.append(a)

def get_pseudo_label(idx,ssv2_best_probs,uni_best_probs,mvit_best_probs, slowfast_best_probs):
    ssv2_probs = ssv2_best_probs
    # print("ssv2_probs",ssv2_probs)
    ssv2_preds = ssv2_probs.argmax(1)
    # print("ssv2_preds",ssv2_preds)

    indx_7_6 = np.bitwise_or((ssv2_preds==6),(ssv2_preds==7))
    # c76 = 0
    # for i in indx_7_6:
    #     if i == True:
    #         c76+=1
    # print("indx_7_6",indx_7_6,c76)
    # cother = 0
    other_indx = np.bitwise_xor(np.ones(len(ssv2_preds)).astype(np.bool),indx_7_6)
    # for i in other_indx:
    #     if i == True:
    #         cother+=1
    # print("other_indx",other_indx,cother)

    new_probs = np.zeros(ssv2_probs.shape)
    uni_probs = uni_best_probs

    # new_probs[other_indx]=ssv2_probs[other_indx]*0.375+uni_last_probs[other_indx]*0.375+uni_best_probs[other_indx]*0.1+mvit_best_probs[other_indx]*0.1+slowfast_best_probs[other_indx]*0.1
    new_probs[other_indx]=ssv2_probs[other_indx]*0.1+uni_probs[other_indx]*0.7+mvit_best_probs[other_indx]*0.1+slowfast_best_probs[other_indx]*0.1
    new_probs[indx_7_6] = ssv2_probs[indx_7_6]*0.8 + uni_probs[indx_7_6]*0.2
    probs = new_probs
    new_preds = new_probs.argmax(1)
    # preds = []
    # idx_new = []
    # probs_new = []
    # # print("new_probs",len(new_probs))
    # # print("new_preds",len(new_preds))
    # # print("idx",len(idx))

    # for i,_ in enumerate(new_probs):
    #     # if max(new_probs[i]) >= label_mean_th_map[new_probs[i].argmax()]:
    #         preds.append(new_preds[i])
    #         idx_new.append(idx[i])
    #         probs_new.append(probs[i])

    # print("preds",len(preds))
    # print("idx_new",len(idx_new))
    # new_preds = new_probs.argmax(1)
    for i,_ in enumerate(new_preds):
        if new_preds[i]==5 or new_preds[i]==9:
            new_probs[i] = slowfast_best_probs[i]

    new_preds = new_probs.argmax(1)

    return new_preds,idx,new_probs





ssv2_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_ssv2_best, 'rb'))['video_preds']).softmax(-1).numpy()#[:3088]
print("ssv2_best_probs",ssv2_best_probs.shape)
# ssv2_last_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_ssv2_last, 'rb'))['video_preds']).softmax(-1).numpy()

uni_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_uni_best, 'rb'))['video_preds']).softmax(-1).numpy()#[:3088]
print("uni_best_probs",uni_best_probs.shape)
# uni_last_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_uni_last, 'rb'))['video_preds']).softmax(-1).numpy()

mvit_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_mvit_best, 'rb'))['video_preds']).softmax(-1).numpy()#[:3088]
print("mvit_best_probs",mvit_best_probs.shape)
slowfast_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_slowfast_best, 'rb'))['video_preds']).softmax(-1).numpy()#[:3088]
print("slowfast_best_probs",slowfast_best_probs.shape)

f=open(args.gen_label_list_path,"r")
idx = f.readlines()
f.close()
idx = [x.strip().split(",")[0] for x in idx]

pred,idx,prob = get_pseudo_label(idx,ssv2_best_probs,uni_best_probs,mvit_best_probs,slowfast_best_probs)

prob = np.array(prob)
# print("prob",type(prob),prob.shape)
# print("prob[:,1]",type(prob),prob[:,3].shape)

# print("idx",len(idx))
# print("pred",len(pred))
# print("prob",len(prob))

# pd.DataFrame({"id":idx,"label":pred}).to_csv(args.path_pseudo_label,index=False,header=False)

pd.DataFrame({"id":idx,"label":pred,"prob0":prob[:,0],"prob1":prob[:,1],"prob2":prob[:,2] \
    ,"prob3":prob[:,3],"prob4":prob[:,4],"prob5":prob[:,5],"prob6":prob[:,6] \
    ,"prob7":prob[:,7],"prob8":prob[:,8],"prob9":prob[:,9],"prob10":prob[:,10]}).to_csv(args.path_pseudo_label,index=False,header=False)


real_csv_list = []
with open('/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/data/real+arid_unlabel.csv')as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        real_csv_list.append(row)
cnt = 0
t_cnt = 0
real = {}
for i,row in enumerate(real_csv_list):
    real[row[0]]=row[1]

real_list = []
for i in range(11):
    real_list.append(0)

for i,row in enumerate(real_csv_list):
    real_list[int(row[1])] +=1   

f_list = []
for i in range(11):
    f_list.append(0)

T_list = []
for i in range(11):
    T_list.append(0)

for i,_ in enumerate(idx):
    cnt += 1
    # print(idx[i],pred[i],type(pred[i]),real[idx[i]],type(real[idx[i]]))
    if int(real[idx[i]]) == int(pred[i]):
        # print("----------------")
        t_cnt+=1
        T_list[pred[i]]+=1
    else:
        f_list[int(real[idx[i]])]+=1

# for i,row in enumerate(real_csv_list):
#     cnt += 1
#     if int(row[1]) == int(pred[i]):
#         label_th_list[pred[i]].append(max(prob[i]))
#         t_cnt+=1

print("total number of pseudo_label = ",cnt)
# print("t_cnt = ",t_cnt)
print("acc = ",t_cnt/cnt)
print("f_list:")
for i in range(11):
    print(i,f_list[i])
print("T_list:")
for i in range(11):
    print(i,T_list[i],real_list[i],T_list[i]/real_list[i])

# for i in range(11):
#     if len(label_th_list[i])!=0:
#         m = np.mean(label_th_list[i])
#     else:
#         m = 0
#     print(i,m,len(label_th_list[i]))


