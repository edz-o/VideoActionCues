from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy,
                                               mean_class_accuracy)
import pickle
import sys

with open(sys.argv[1], 'rb' ) as f:
    results, gt_labels = pickle.load(f)

top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
mean_acc = mean_class_accuracy(results, gt_labels)
print("Mean Class Accuracy = {:.02f}".format(mean_acc * 100))
print("Top-1 Accuracy = {:.02f}".format(top1 * 100))
print("Top-5 Accuracy = {:.02f}".format(top5 * 100))
