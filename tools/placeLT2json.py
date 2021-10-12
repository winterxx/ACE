import os,re
import json
from PIL import Image
import numpy as np


train_image_root = '/home/jrcai/hd/OLTR/Places_LT_v2/train_256_places365standard/'
test_image_root = '/home/jrcai/hd/OLTR/Places_LT_v2/'
val_image_root = '/home/jrcai/hd/OLTR/Places_LT_v2/train_256_places365standard/'
open_image_root = '/home/jrcai/hd/OLTR/Places_LT_v2/Places_LT_open/'

train_label_root = '/home/jrcai/hd/OLTR/Places_LT_v2/Places_LT_train.txt'
test_label_root = '/home/jrcai/hd/OLTR/Places_LT_v2/Places_LT_test.txt'
val_label_root = '/home/jrcai/hd/OLTR/Places_LT_v2/Places_LT_val.txt'
open_label_root = '/home/jrcai/hd/OLTR/Places_LT_v2/Places_LT_open.txt'

train_output = './data/Places_train.json'
test_output = './data/Places_test.json'
val_output = './data/Places_val.json'
open_output = './data/Places_open.json'
label_output = './data/Places_label.json'

label_root_list = [train_label_root,test_label_root,val_label_root,open_label_root]
image_root_list = [train_image_root,test_image_root,val_image_root,open_image_root]
output_list = [train_output,test_output,val_output,open_output]
label_list = [""]*365
num_class_list = [0 for i in range(365)]
label_root = train_label_root

order2sorted = {}
with open('/home/jrcai/hd/OLTR/Places_LT_v2/Places_LT_train.txt', 'r') as f:
    for line in f.readlines():
        path, cls = line.strip().split()
        num_class_list[int(cls)] += 1
num_class_list = np.asarray(num_class_list)
ranked = np.argsort(num_class_list)
largest_indices = ranked[::-1].copy()
num_class_list = np.asarray(num_class_list)[largest_indices]

cnt = 0
for idx in largest_indices:
    order2sorted[idx] = cnt
    cnt += 1

with open(label_root, 'r') as f:
    for line in f:
        filename, cls = line.split()
        path = os.path.normpath(filename)
        cls_name = path.split(os.sep)[-2]
        cls = order2sorted[int(cls)]
        label_list[cls] = cls_name
with open(label_output, 'w') as outfile:
    json.dump(label_list, outfile)

for i in range(len(label_root_list)):
    label_root, image_root, output = label_root_list[i], image_root_list[i], output_list[i]
    anno_list = []
    img_id = 0
    with open(label_root,'r') as f:
        for line in f:
            filename, cls = line.split()
            cls = order2sorted[int(cls)]
            #if 'train' in output or 'val' in output:
            #    filename = filename[6:]
            #else:
            #filename = os.path.basename(filename)
            img_path = os.path.join(image_root, filename)
            image = Image.open(img_path)
            width, height = image.size
            anno = {'image_id': img_id,
                    'fpath': img_path,
                    'im_height': height,
                    'im_width': width,
                    'category_id': int(cls)}
            anno_list.append(anno)
            img_id += 1
    jsonfile = {
        "annotations": anno_list,
        'num_classes': 365
    }
    print(label_root, len(anno_list))
    with open(output, "w") as outfile:
        json.dump(jsonfile, outfile, indent=4)

