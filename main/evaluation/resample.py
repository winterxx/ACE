import os,re
import json
import operator
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
from scipy.stats import entropy
def mean( hist ):
    mean = 0.0;
    for i in hist:
        mean += i;
    mean/= len(hist);
    return mean;

def bhatta ( hist1,  hist2):
    # calculate mean of hist1
    h1_ = mean(hist1);

    # calculate mean of hist2
    h2_ = mean(hist2);

    # calculate score
    score = 0;
    for i in range(100):
        score += math.sqrt( hist1[i] * hist2[i] );
    # print h1_,h2_,score;
    score = math.sqrt( 1 - ( 1 / math.sqrt(h1_*h2_*100*100) ) * score );
    return score;


train_json_file = '/home/jrcai/hd/OLTR/CIFAR/converted/cifar100_imbalance100/cifar100_imbalance100_train.json'
with open(train_json_file, 'r') as f:
    data = json.load(f)
train_anno = data['annotations']
num_class = data['num_classes']

train_num_sample = {}
for sample in train_anno:
    cid = sample['category_id']
    if cid not in train_num_sample.keys():
        train_num_sample[cid] = 0
    train_num_sample[cid] += 1

train_num_sample = dict(sorted(train_num_sample.items(), key=operator.itemgetter(1),reverse=True))
print(train_num_sample.values())
train_dist = list(train_num_sample.values())
train_max = sum(train_dist)
train_norm = [x/train_max for x in train_dist]
"""
test_json_file = '/home/jrcai/hd/OLTR/CIFAR/converted/cifar100_imbalance100/cifar100_imbalance100_valid.json'
with open(test_json_file, 'r') as f:
    data = json.load(f)
test_anno = data['annotations']

test_num_sample = {}
test_sample = {}
for sample in train_anno:
    cid = sample['category_id']
    fpath = sample['fpath']
    if cid not in test_sample.keys():
        test_sample[cid] = []
    test_sample[cid].append(fpath)

a, m = 6., 2.
s_list = []
for j in range(200):
    s = (np.random.pareto(a, 100) + 1) * m
    s_list.append(bhatta(s,train_dist))
print(s_list)

plt.figure()
plt.plot(list(range(100)),y1,label='y1')
plt.plot(list(range(100)),y2,label='y2')
plt.plot(list(range(100)),y3,label='y3')
plt.plot(list(range(100)),y4,label='y4')
plt.plot(list(range(100)),y5,label='y5')
plt.legend()
plt.show()
"""
s_list = []
plt.figure()
plt.plot(list(range(100)), train_norm)
for j in range(5):
    permutation = np.random.permutation(100)
    freq_list = [0.01**(x/99) for x in permutation]
    test_sum = sum(freq_list)
    freq_list = [x/test_sum for x in freq_list]
    plt.plot(list(range(100)), freq_list)
    s_list.append(bhatta(freq_list,train_norm))
print(sorted(s_list))
plt.show()