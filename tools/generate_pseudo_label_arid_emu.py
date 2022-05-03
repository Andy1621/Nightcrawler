# from winreg import EnumValue
import torch
import pickle
import csv
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gen_label_list_path', type=str, default="/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/data/arid_unlabel.csv")
parser.add_argument('--gen_label_pkl_path_ssv2_best', type=str, default="/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/exp_adapt_bn/uniformer_b32_ssv2/real_unlabelx32x224x1x1.pkl")
# parser.add_argument('--gen_label_pkl_path_ssv2_last', type=str, default="/mnt/lustre/shangjingjie1/newexp/uniformer_k400_ssv2_bn/last_epoch/pseudo_train_unlabelx32x224x1x1.pkl")
parser.add_argument('--gen_label_pkl_path_uni_best', type=str, default="/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/exp_adapt_bn/uniformer_b32_k600/real_unlabelx32x224x1x1.pkl")
# parser.add_argument('--gen_label_pkl_path_uni_last', type=str, default="/mnt/lustre/shangjingjie1/newexp/uniformer_bn2/last_epoch/pseudo_train_unlabelx32x224x1x1.pkl")
parser.add_argument('--gen_label_pkl_path_mvit_best', type=str, default="/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/exp_adapt_bn/mvit_b32_k600_dp0.3/real_unlabelx32x224x1x1.pkl")
parser.add_argument('--gen_label_pkl_path_slowfast_best', type=str, default="/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/exp_adapt_bn/sf32_k600/real_unlabelx32x224x1x1.pkl")
parser.add_argument('--path_pseudo_label', type=str, default="/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/data/pseudo/test.csv")
args = parser.parse_args()

label_min_th_map = {0:0.247,1:0.178,2:0.236,3:0.207,4:0.175,5:0.202,6:0.219,7:0.246 \
                    ,8:0.184,9:0.211,10:0.202}

label_mean_th_map = {0:0.554,1:0.418,2:0.384,3:0.425,4:0.388,5:0.504,6:0.772,7:0.733 \
                    ,8:0.494,9:0.582,10:0.590}

# label_mean_th_map = {0:0.7,1:0.418,2:0.384,3:0.425,4:0.388,5:0.68,6:0.772,7:0.8 \
#                     ,8:0.6,9:0.66,10:0.590}


label_th_list = []
for i in range(11):
    a = []
    label_th_list.append(a)

def get_pseudo_label(cla,th,idx,ssv2_best_probs,uni_best_probs,mvit_best_probs,slowfast_best_probs):
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
    new_probs[other_indx]=ssv2_probs[other_indx]*0.375+uni_probs[other_indx]*0.375+mvit_best_probs[other_indx]*0.1+slowfast_best_probs[other_indx]*0.1
    new_probs[indx_7_6] = ssv2_probs[indx_7_6]*0.8 + uni_probs[indx_7_6]*0.2
    probs = new_probs
    new_preds = new_probs.argmax(1)
    preds = []
    idx_new = []
    probs_new = []
    # print("new_probs",len(new_probs))
    # print("new_preds",len(new_preds))
    # print("idx",len(idx))

    for i,_ in enumerate(new_probs):
        if new_probs[i].argmax() == cla and max(new_probs[i]) >= th:
            preds.append(new_preds[i])
            idx_new.append(idx[i])
            probs_new.append(probs[i])

    # preds = new_preds
    # idx_new = idx
    # probs_new = probs
    # print("preds",len(preds))
    # print("idx_new",len(idx_new))

    return preds,idx_new,probs_new





ssv2_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_ssv2_best, 'rb'))['video_preds']).softmax(-1).numpy()[3088:]
print("ssv2_best_probs",ssv2_best_probs.shape)
# ssv2_last_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_ssv2_last, 'rb'))['video_preds']).softmax(-1).numpy()

uni_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_uni_best, 'rb'))['video_preds']).softmax(-1).numpy()[3088:]
print("uni_best_probs",uni_best_probs.shape)
# uni_last_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_uni_last, 'rb'))['video_preds']).softmax(-1).numpy()

mvit_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_mvit_best, 'rb'))['video_preds']).softmax(-1).numpy()[3088:]
print("mvit_best_probs",mvit_best_probs.shape)
slowfast_best_probs = torch.Tensor(pickle.load(open(args.gen_label_pkl_path_slowfast_best, 'rb'))['video_preds']).softmax(-1).numpy()[3088:]
print("slowfast_best_probs",slowfast_best_probs.shape)

f=open(args.gen_label_list_path,"r")
idx = f.readlines()
f.close()
idx = [x.strip().split(",")[0] for x in idx]

real_csv_list = []
with open('/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/data/real_arid_unlabel.csv')as f:
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
    # print("prob",type(prob),prob.shape)
    # print("prob[:,1]",type(prob),prob[:,3].shape)

    # print("idx",len(idx))
    # print("label",len(pred))

    # pd.DataFrame({"id":idx,"label":pred}).to_csv(args.path_pseudo_label,index=False,header=False)

    # pd.DataFrame({"id":idx,"label":pred,"prob0":prob[:,0],"prob1":prob[:,1],"prob2":prob[:,2] \
    #     ,"prob3":prob[:,3],"prob4":prob[:,4],"prob5":prob[:,5],"prob6":prob[:,6] \
    #     ,"prob7":prob[:,7],"prob8":prob[:,8],"prob9":prob[:,9],"prob10":prob[:,10]}).to_csv(args.path_pseudo_label,index=False,header=False)


    
    # cnt = 0
    t_cnt = 0


    # f_list = []
    # for i in range(11):
    #     f_list.append(0)


    for i,_ in enumerate(idx_x):
        # cnt += 1
        # print(idx_x[i],real[idx[i]])
        # print(int(cla))
        # if int(real[idx_x[i]]) == int(cla):
            # cnt+=1
        # print(idx_x[i],real[idx_x[i]],pred[i])
        if int(real[idx_x[i]]) == int(pred[i]):
            # print("----------------")
            t_cnt+=1
        # else:
        #     f_list[pred[i]]+=1

    # print("idx",len(idx))
    # print("pred",len(pred))

    # for i,row in enumerate(real_csv_list):
    #     cnt += 1
    #     if int(row[1]) == int(pred[i]):
    #         label_th_list[pred[i]].append(max(prob[i]))
    #         t_cnt+=1
    # ratio = 0
    ratio = t_cnt/len(pred)
    # print("ra
    # tio",ratio)
    f1 = (2*(len(pred)/real_label_cnt[cla])*(t_cnt/len(pred)))/(len(pred)/real_label_cnt[cla])+(t_cnt/len(pred))
    if f1>maxx:
        maxx=f1
        print("th:",th)
        print("s",len(pred))
        print("total number of real_label = ",t_cnt)
        print("acc = ",t_cnt/len(pred))
        print("acc2 = ",len(pred)/real_label_cnt[cla])
        print("F1",f1)

    # if ratio>0.9 and ratio!=1:
        
# for i in range(11):
#     print(i,f_list[i])
# for i in range(11):
#     if len(label_th_list[i])!=0:
#         m = np.mean(label_th_list[i])
#     else:
#         m = 0
#     print(i,m,len(label_th_list[i]))


