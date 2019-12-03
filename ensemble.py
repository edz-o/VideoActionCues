
# coding: utf-8

import pickle
import numpy as np

from mmaction.core.evaluation.accuracy import top_k_accuracy

def load_pkl(fname):
    with open(fname, 'rb') as f:
        results, gt_labels, video_names = pickle.load(f)
    return results, gt_labels, video_names


fname_real_only = 'results/real_only.pkl'
fname_optical_only = 'results/flow_only.pkl'
fname_kp = 'results/rgb+kp.pkl'
fname_sim = 'results/sim_augmentation.pkl'

fnames = [fname_real_only, fname_optical_only, fname_kp, fname_sim]

# Choose modalities to ensemble
ensemble_ids = [0,1,2,3]

all_res = []
for i in ensemble_ids:
    res, gt, vn = load_pkl(fnames[i])

    order = np.argsort(vn)
    res = np.array(res)
    gt = np.array(gt)

    res = res[order, :]
    gt = gt[order]
    all_res.append(res)

res = np.array(all_res).sum(0)
res = [logits for logits in res]
gt = gt.tolist()


top1, top5 = top_k_accuracy(res, gt, k=(1, 5))
print("Top-1 Accuracy = {:.02f}".format(top1 * 100))

