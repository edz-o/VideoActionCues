import pickle, json, csv
from sklearn.metrics import confusion_matrix
import numpy as np
import sys

result = pickle.load(open(sys.argv[1], 'rb'))
pred = [res.argmax() for res in result[0] ]
cfm = confusion_matrix(result[1], pred, labels=list(range(400)))
kinetics_mapping = json.load(open('/data/yzhang/mmaction/data/kinetics400/kinetics_class_mapping.json'))
kinetics_mapping_reverse = { v:k for k, v in kinetics_mapping.items()}
labels = [kinetics_mapping_reverse[i] for i in range(400)]

#for i in range(len(pred)):
#    print('pred: %s label: %s' % (kinetics_mapping_reverse[ pred[i]], kinetics_mapping_reverse[result[1][i]]))

for row in range(cfm.shape[0]):
    if cfm[row].sum() > 0:
        #print(kinetics_mapping_reverse[row])
        top_5_preds = np.argsort(cfm[row])[::-1][:5]
        #print('Prediction:')
        for cl in top_5_preds:
            sys.stdout.write('%s (%.3f),' % (kinetics_mapping_reverse[cl], cfm[row][cl] / cfm[row].sum()))
        sys.stdout.write('\n')
        sys.stdout.flush()

for row in range(cfm.shape[0]):
    if cfm[row].sum() > 0:
        print("%.3f" % (cfm[row][row] / cfm[row].sum()))
