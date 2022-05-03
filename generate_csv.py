import torch
import pickle
import csv
import slowfast.utils.metrics as metrics

pre_path = '/mnt/lustre/likunchang.vendor/sjj/ug2_uniformer_competition/exp_pseudo_arid_stage4/uniformer_b32_ssv2_ce/testx32x224x1x3.pkl'

test_path = 'data/test.csv'

pred = pickle.load(open(pre_path, 'rb'))
pred_score = torch.Tensor(pred['video_preds']).softmax(-1)
pred_label = pred_score.argmax(-1)
pre_set = set()
print(pred_label)

print(torch.Tensor(pred['video_labels']).long())

# truth
ks = (1, 5)
res_labels = torch.from_numpy(pred['video_labels'])
num_topks_correct = metrics.topks_correct(
        pred_score, res_labels, ks
    )
topks = [
    (x / pred_score.size(0)) * 100.0
    for x in num_topks_correct
]
assert len({len(ks), len(topks)}) == 1
for k, topk in zip(ks, topks):
    print(
        "top{}_acc".format(k), 
        "{:.{prec}f}".format(topk, prec=2)
    )

with open(test_path, 'r') as f:
    test_lines = f.readlines()

out = open("arid_pred.csv", "w", newline = "")
csv_writer = csv.writer(out, dialect = "excel")
csv_writer.writerow(['VideoID', 'Video', 'ClassID', 'Probability'])

for i in range(len(test_lines)):
    file_name = test_lines[i].rstrip().split(',')[0]
    csv_writer.writerow([i, file_name, pred_label[i].item(), pred_score[i].unsqueeze(0)])

print('ok')