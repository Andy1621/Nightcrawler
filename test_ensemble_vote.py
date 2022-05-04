import pickle
import torch
import slowfast.utils.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import os


def get_probs(pkl_path):
    with open(pkl_path, 'rb') as f:
        res = pickle.load(f)
    
    res_labels = torch.from_numpy(res['video_labels']).numpy()
    res_logits = torch.from_numpy(res['video_preds'])
    res_probs = res_logits.softmax(1).numpy()
    return res_labels,res_probs


def topk(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort


def get_em_prob(ssv2_best_probs,uni_best_probs,mvit_best_probs, slowfast_best_probs,dense_best_probs,mode):
    ssv2_probs = ssv2_best_probs
    ssv2_preds = ssv2_probs.argmax(1)
    indx_7_6 = np.bitwise_or((ssv2_preds==6),(ssv2_preds==7))
    other_indx = np.bitwise_xor(np.ones(len(ssv2_preds)).astype(np.bool),indx_7_6)
    new_probs = np.zeros(ssv2_probs.shape)
    uni_probs = uni_best_probs
    if mode == 0 or mode == 1:
        new_probs[other_indx]=ssv2_probs[other_indx]*0.6+uni_probs[other_indx]*0.1+mvit_best_probs[other_indx]*0.1+slowfast_best_probs[other_indx]*0.1+dense_best_probs[other_indx]*0.1
    elif mode == 2:
        new_probs[other_indx]=ssv2_probs[other_indx]*0.4+uni_probs[other_indx]*0.3+mvit_best_probs[other_indx]*0.1+slowfast_best_probs[other_indx]*0.1+dense_best_probs[other_indx]*0.1
    elif mode == 3 or mode == 4:
        new_probs[other_indx]=ssv2_probs[other_indx]*0.2+uni_probs[other_indx]*0.2+mvit_best_probs[other_indx]*0.2+slowfast_best_probs[other_indx]*0.2+dense_best_probs[other_indx]*0.2
    
    new_probs[indx_7_6] = ssv2_probs[indx_7_6]*0.1 + uni_probs[indx_7_6]*0.2

    new_preds = new_probs.argmax(1)

    for i,_ in enumerate(new_preds):
        if new_preds[i]==5 or new_preds[i]==9:
            new_probs[i] = slowfast_best_probs[i]

    return new_probs


def get_em_prob_4(ssv2_best_probs,uni_best_probs,mvit_best_probs, slowfast_best_probs):
    ssv2_probs = ssv2_best_probs
    ssv2_preds = ssv2_probs.argmax(1)
    indx_7_6 = np.bitwise_or((ssv2_preds==6),(ssv2_preds==7))
    other_indx = np.bitwise_xor(np.ones(len(ssv2_preds)).astype(np.bool),indx_7_6)
    new_probs = np.zeros(ssv2_probs.shape)
    uni_probs = uni_best_probs
    new_probs[other_indx]=ssv2_probs[other_indx]*0.25+uni_probs[other_indx]*0.25+mvit_best_probs[other_indx]*0.25+slowfast_best_probs[other_indx]*0.25
    new_probs[indx_7_6] = ssv2_probs[indx_7_6]*0.8 + uni_probs[indx_7_6]*0.2

    new_preds = new_probs.argmax(1)

    for i,_ in enumerate(new_preds):
        if new_preds[i]==5 or new_preds[i]==9:
            new_probs[i] = slowfast_best_probs[i]

    return new_probs


def deal_preds(preds,probs):
    for i in range(len(preds)):
        if preds[i]==10 and probs[i][10]<=0.84:
            if (probs[i][2]>=0.2):  #0.28
                preds[i] = 2
            elif( probs[i][3]>=0.1): # 0.10
                preds[i] = 3
            elif (probs[i][1]>=0.1): #20
                preds[i] = 1
            elif (probs[i][8]>=0.2): #17  0.03
                preds[i] = 8
            elif (probs[i][0]>=0.01): #17 0.03
                preds[i] = 0

    for i in range(len(preds)):
        if preds[i]==9 and probs[i][9]<=0.94:
            if( probs[i][5]>=0.2): #0.12
                preds[i] = 5
            elif( probs[i][8]>=0.1): #0.1
                preds[i] = 8
    
    for i in range(len(preds)):
        if preds[i]==8 and probs[i][9]<=0.73:  #bug
            if( probs[i][1]>=0.2): #0.32
                preds[i] = 1
            elif( probs[i][4]>=0.1): #.0.1
                preds[i] = 4
            elif( probs[i][0]>=0.1): #0.08
                preds[i] = 0
            elif( probs[i][10]>=0.7): #0.4
                preds[i] = 10

    for i in range(len(preds)):
        if preds[i]==7 and probs[i][7]<=0.26:
            if( probs[i][2]>=0.05): #0.02
                preds[i] = 2

    ## for i in range(len(preds)):
    
    ##     if preds[i]==5 and probs[i][5]<=0.6:
    ##         if( probs[i][9]>=0.05):
    ##             preds[i] = 9
    ##         if( probs[i][8]>=0.01):
    ##             preds[i] = 1

    for i in range(len(preds)):
        if preds[i]==4 and probs[i][4]<=0.83:
            if( probs[i][5]>=0.2): #0.11
                preds[i] = 5
            elif (probs[i][1]>=0.1): #0.05
                preds[i] = 1
            elif (probs[i][10]>=0.1): #0.05
                preds[i] = 10
            elif (probs[i][3]>=0.05): #0.05
                preds[i] = 3
    
    for i in range(len(preds)):
        if preds[i]==3 and probs[i][3]<=0.7:
            if( probs[i][2]>=0.2): #0.11
                preds[i] = 2
            elif (probs[i][0]>=0.03): #0.05
                preds[i] = 0

    # for i in range(len(preds)):
    #     if preds[i]==2 and probs[i][2]<=0.68:
    #         if( probs[i][3]>=0.5): #0.11
    #             preds[i] = 3

    # find_idx = 8
    # idx_walk  = (preds==find_idx)
    # probs_walk = probs[idx_walk]
    # labels_walk = labels[idx_walk]

    # M = {}
    # other = []
    # for label_walk,prob_walk in zip(labels_walk,probs_walk):
    #     if(label_walk!=find_idx):
    #         other.append(prob_walk[find_idx])
    #     if label_walk not in M:
    #         M[label_walk] = []
    #         M[label_walk].append(prob_walk[label_walk])
    #     else:
    #         M[label_walk].append(prob_walk[label_walk])
    
    # for k,v in M.items():
    #     print(k,np.mean(v))
    # print(np.mean(other))
    return preds


def deal_preds_v2(preds,probs):
    for i in range(len(preds)):
        if preds[i]==4 and probs[i][4]<=0.83:
            if( probs[i][5]>=0.1): #0.12
                preds[i] = 5
            elif (probs[i][1]>=0.3): #0.10
                preds[i] = 1
            elif (probs[i][3]>=0.03): #0.03
                preds[i] = 3
    
    for i in range(len(preds)):
        if preds[i]==7 and probs[i][7]<=0.26:
            if( probs[i][2]>=0.05): #0.02
                preds[i] = 2
    
    for i in range(len(preds)):
        if preds[i]==8 and probs[i][8]<=0.73:
            if( probs[i][10]>=0.2): #0.32
                preds[i] = 10
            elif( probs[i][4]>=0.3): #0.22
                preds[i] = 4
            elif( probs[i][0]>=0.1): #0.06
                preds[i] = 0

    for i in range(len(preds)):
        if preds[i]==9 and probs[i][9]<=0.94:
            if( probs[i][5]>=0.3): #0.12
                preds[i] = 5
            elif( probs[i][8]>=0.05): #0.1
                preds[i] = 8
    
    for i in range(len(preds)):
        if preds[i]==10 and probs[i][10]<=0.84:
            if (probs[i][9]>=0.01):  #0.01
                preds[i] = 9
            elif( probs[i][3]>=0.07): # 0.10
                preds[i] = 3
            elif (probs[i][1]>=0.1): #20
                preds[i] = 1
            elif (probs[i][0]>=0.008): #17 0.03
                preds[i] = 0

    # find_idx = 10
    # idx_walk  = (preds==find_idx)
    # probs_walk = probs[idx_walk]
    # labels_walk = labels[idx_walk]

    # M = {}
    # other = []
    # for label_walk,prob_walk in zip(labels_walk,probs_walk):
    #     if(label_walk!=find_idx):
    #         other.append(prob_walk[find_idx])
    #     if label_walk not in M:
    #         M[label_walk] = []
    #         M[label_walk].append(prob_walk[label_walk])
    #     else:
    #         M[label_walk].append(prob_walk[label_walk])
    
    # for k,v in M.items():
    #     print(k,np.mean(v))
    # print(np.mean(other))

    return preds


def deal_preds_v3(preds,probs):
    for i in range(len(preds)):
        if preds[i]==10 and probs[i][10]<=0.8:
            if( probs[i][3]>=0.15): # 0.10
                preds[i] = 3
            elif (probs[i][1]>=0.05): #15
                preds[i] = 1
            elif (probs[i][8]>=0.3): #17  0.03
                preds[i] = 8
            elif (probs[i][0]>=0.01): #17 0.06
                preds[i] = 0
    
    for i in range(len(preds)):
        if preds[i]==8 and probs[i][8]<=0.8:  #bug
            if( probs[i][4]>=0.05): #.0.1
                preds[i] = 4
            elif( probs[i][0]>=0.2): #0.08
                preds[i] = 0

    # for i in range(len(preds)):
    #     if preds[i]==7 and probs[i][7]<=0.26:
    #         if( probs[i][2]>=0.05): #0.02
    #             preds[i] = 2    
    
    for i in range(len(preds)):
        if preds[i]==4 and probs[i][4]<=0.83:
            if( probs[i][5]>=0.2): #0.11
                preds[i] = 5
            elif (probs[i][1]>=0.2): #0.05
                preds[i] = 1
            elif (probs[i][3]>=0.1): #0.005
                preds[i] = 3
    
    for i in range(len(preds)):
        if preds[i]==3 and probs[i][3]<=0.7:
            if (probs[i][0]>=0.01): #0.05
                preds[i] = 0
    
    for i in range(len(preds)):
        if preds[i]==9 and probs[i][9]<=0.96:
            if( probs[i][4]>=0.2): #0.01
                preds[i] = 4
            # elif( probs[i][4]>=0.1): #0.01
            #     preds[i] = 4
            # elif( probs[i][8]>=0.1): #0.01
            #     preds[i] = 8
    return preds


def deal_preds_v4(preds,probs):
    for i in range(len(preds)):
        if preds[i]==10 and probs[i][10]<=0.9:
            if( probs[i][3]>=0.2): # 0.10
                preds[i] = 3
            elif (probs[i][1]>=0.05): #15
                preds[i] = 1
            elif (probs[i][8]>=0.3): #17  0.03
                preds[i] = 8
            elif (probs[i][0]>=0.008): #17 0.06
                preds[i] = 0
    
    for i in range(len(preds)):
        if preds[i]==8 and probs[i][8]<=0.8:  #bug
            if( probs[i][4]>=0.05): #.0.1
                preds[i] = 4
            elif( probs[i][0]>=0.2): #0.08
                preds[i] = 0

    # for i in range(len(preds)):
    #     if preds[i]==7 and probs[i][7]<=0.26:
    #         if( probs[i][2]>=0.05): #0.02
    #             preds[i] = 2
            
    for i in range(len(preds)):
        if preds[i]==4 and probs[i][4]<=0.9:
            if( probs[i][5]>=0.2): #0.11
                preds[i] = 5
            elif (probs[i][1]>=0.2): #0.05
                preds[i] = 1
            elif (probs[i][3]>=0.2): #0.005
                preds[i] = 3
    
    for i in range(len(preds)):
        if preds[i]==3 and probs[i][3]<=0.7:
            if (probs[i][0]>=0.001): #0.05
                preds[i] = 0
    
    for i in range(len(preds)):
        if preds[i]==9 and probs[i][9]<=0.8:
            if( probs[i][4]>=0.01): #0.01
                preds[i] = 4
            # elif( probs[i][4]>=0.1): #0.01
            #     preds[i] = 4
            # elif( probs[i][8]>=0.1): #0.01
            #     preds[i] = 8

    return preds


def select_thres(mode):
    if mode == 0 or mode == 1 or mode == 2:
        ssv2_best_path = "././exp_pseudo_stage4/uniformer_b32_ssv2/testx32x224x1x3.pkl"
        uni_best_path = "./exp_pseudo_stage4/uniformer_b32_k600/testx32x224x1x3.pkl"
        mvit_best_path = "./exp_pseudo_stage4/mvit_b32_k600/testx32x224x1x3.pkl"
        slowfast_best_path = "./exp_pseudo_stage4/sf32_k700/testx32x224x1x3.pkl"
        dense_best_path = "./exp_experts/uniformer_b32_ssv2/testx32x224x1x1.pkl"
    elif mode == 3 or mode == 4:
        ssv2_best_path = "./exp_pseudo_stage4/uniformer_b32_ssv2/testx32x224x3x3.pkl"
        uni_best_path = "./exp_pseudo_stage4/uniformer_b32_k600/testx32x224x3x3.pkl"
        mvit_best_path = "./exp_pseudo_stage4/mvit_b32_k600/testx32x224x3x3.pkl"
        slowfast_best_path = "./exp_pseudo_stage4/sf32_k700/testx32x224x3x3.pkl"
        dense_best_path = "./exp_experts/uniformer_b32_ssv2/testx32x224x3x3.pkl"

    _, ssv2_best_probs = get_probs(ssv2_best_path)
    _, uni_best_probs = get_probs(uni_best_path)
    _, mvit_best_probs = get_probs(mvit_best_path)
    _, slowfast_best_probs = get_probs(slowfast_best_path)
    _, dense_best_probs = get_probs(dense_best_path)

    probs = get_em_prob(ssv2_best_probs, uni_best_probs, mvit_best_probs, slowfast_best_probs,dense_best_probs,mode)
    
    # probs = ssv2_best_probs
    preds = probs.argmax(1)

    if mode == 1:
        preds = deal_preds(preds,probs)
    elif mode == 2:
        preds = deal_preds_v2(preds,probs)
    elif mode == 3:
        preds = deal_preds_v3(preds,probs)
    elif mode == 4:
        preds = deal_preds_v4(preds,probs)

    probs = torch.from_numpy(probs).softmax(1).to(torch.float32)

    return preds,probs

    # print(confusion_matrix(labels, newpreds))
    # print(accuracy_score(labels, newpreds))


def generate_final_sub(preds,probs,mode):
    test_path = 'data/test.csv'
    with open(test_path, 'r') as f:
        test_lines = f.readlines()

    out = open("submission/arid_pred_{}.csv".format(mode), "w", newline = "")
    csv_writer = csv.writer(out, dialect = "excel")
    csv_writer.writerow(['VideoID', 'Video', 'ClassID', 'Probability'])

    for i in range(len(test_lines)):
        file_name = test_lines[i].rstrip().split(',')[0]
        csv_writer.writerow([i, file_name, preds[i], probs[i].unsqueeze(0).data])


def vote_for_sub():
    prefix = './submission'

    file_list = [
        # 'arid_pred_0.csv', 
        'arid_pred_1.csv', 
        # 'arid_pred_2.csv', 
        # 'arid_pred_3.csv', 
        'arid_pred_4.csv', # best
    ]

    pred_data = []
    for file in file_list:
        with open(os.path.join(prefix, file)) as csvfile:
            pred = []
            csv_reader = csv.reader(csvfile)
            birth_header = next(csv_reader)
            for row in csv_reader:
                pred.append(row)
            pred_data.append(pred)

    vote_pred = []
    for i in range(len(pred_data[0])):
        pred = []
        for j in range(len(pred_data)):
            pred.append(int(pred_data[j][i][2]))
        most_label = np.argmax(np.bincount(np.array(pred))) # most frequent
        vote_pred.append(most_label)

    out = open("arid_pred.csv", "w", newline = "")
    csv_writer = csv.writer(out, dialect = "excel")
    csv_writer.writerow(['VideoID', 'Video', 'ClassID', 'Probability'])

    for i, label in enumerate(vote_pred):
        for j in range(len(pred_data)):
            if int(pred_data[j][i][2]) == label:
                tmp = pred_data[j][i]
                break
        csv_writer.writerow(tmp)


if __name__ == '__main__':
    for i in range(4):
        preds,probs = select_thres(mode=i)
        generate_final_sub(preds,probs,mode=i)
    vote_for_sub()
    