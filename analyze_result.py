import pickle, json, csv
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import pdb

def convert_label(s):
    return s.replace('"', '').replace(' ', '_').replace('(', '-').replace(')', '-')
video_list = json.load(open('data/nturgbd/NTU_RGBD_all.json'))

kinetics_mapping = json.load(open('data/kinetics400/kinetics_class_mapping.json'))
csv_reader = csv.reader(open('data/nturgbd/kinetics_ntu_mapping.csv'))

next(csv_reader)
ntu2kinetics = {}
kinetics2ntuname = {}
for kinetics_name, ntu_name, label in csv_reader:
    kinetics_name = convert_label(kinetics_name)
    assert kinetics_name in kinetics_mapping
    kinetics_label = kinetics_mapping[kinetics_name]
    ntu2kinetics[ int(label) ] =  kinetics_label
    kinetics2ntuname[int(kinetics_label)] = ntu_name

new_mapping = {}
for i, kinetics_id in enumerate(kinetics2ntuname.keys()):
    new_mapping[i] = kinetics2ntuname[kinetics_id]


result = pickle.load(open(sys.argv[1], 'rb'))
pred = [res.argmax() for res in result[0] ]
cfm = confusion_matrix(result[1], pred, labels=list(range(400)))

kinetics_ids = np.array(list(kinetics2ntuname.keys()))
ntu_names = [ kinetics2ntuname[kid] for kid in kinetics_ids ]
cfm = cfm[kinetics_ids, :][:, kinetics_ids]

df_cm = pd.DataFrame(cfm, index = ntu_names,
                          columns = ntu_names)
plt.figure(figsize = (10,6))
sns.heatmap(df_cm, annot=True)
plt.xlabel(' ')
plt.ylabel(' ')
plt.savefig('sim_only_cfm.png')
plt.show()
#for i in range(len(pred)):
#    print('pred: %s label: %s' % (kinetics_mapping_reverse[ pred[i]], kinetics_mapping_reverse[result[1][i]]))

#kinetics_mapping = json.load(open('/data/yzhang/mmaction/data/kinetics400/kinetics_class_mapping.json'))
#kinetics_mapping_reverse = { v:k for k, v in kinetics_mapping.items()}
#labels = [kinetics_mapping_reverse[i] for i in range(400)]
#for row in range(cfm.shape[0]):
#    if cfm[row].sum() > 0:
#        #print(kinetics_mapping_reverse[row])
#        top_5_preds = np.argsort(cfm[row])[::-1][:5]
#        #print('Prediction:')
#        for cl in top_5_preds:
#            sys.stdout.write('%s (%.3f),' % (kinetics_mapping_reverse[cl], cfm[row][cl] / cfm[row].sum()))
#        sys.stdout.write('\n')
#        sys.stdout.flush()
#
#for row in range(cfm.shape[0]):
#    if cfm[row].sum() > 0:
#        print("%.3f" % (cfm[row][row] / cfm[row].sum()))
