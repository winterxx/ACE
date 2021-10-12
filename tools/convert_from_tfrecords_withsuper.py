import tensorflow as tf
import cv2, os, json
import numpy as np
from tqdm import tqdm
import itertools
import argparse
"""
hierarchical = {'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                 'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                 'flowers':	['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                 'food_containers':['bottle', 'bowl', 'can', 'cup', 'plate'],
                'fruit_and_vegetables':['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                'household_electrical_devices':	['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                'household_furniture':['bed', 'chair', 'couch', 'table', 'wardrobe'],
                'insects':['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                'large_carnivores':	['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                'large_man-made_outdoor_things':['bridge', 'castle', 'house', 'road', 'skyscraper'],
                'large_natural_outdoor_scenes':	['cloud', 'forest', 'mountain', 'plain', 'sea'],
                'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                'medium_mammals':['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                'non-insect_invertebrates':['crab', 'lobster', 'snail', 'spider', 'worm'],
                'people':['baby', 'boy', 'girl', 'man', 'woman'],
                'reptiles':['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                'small_mammals':['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                'trees':['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                'vehicles_1':['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                'vehicles_2':['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']}
"""
hierarchical = {'flowers':	['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                'natural_scenes': ['cloud', 'mountain', 'plain', 'sea'],
                'food_containers':['bottle', 'bowl', 'can', 'cup', 'plate'],
                'fruit_and_vegetables':['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                'vegetation_trees': ['forest', 'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                'vehicles':['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train','lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
                'large_man-made_outdoor_things':['bridge', 'castle', 'house', 'road', 'skyscraper'],
                'household':['clock', 'keyboard', 'lamp', 'telephone', 'television','bed', 'chair', 'couch', 'table', 'wardrobe'],
                'insects':['beetle', 'butterfly', 'caterpillar', 'cockroach','bee',
                           'crab', 'lobster', 'snail', 'spider', 'worm','crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                'fish' : [ 'whale','aquarium_fish', 'flatfish', 'ray', 'shark', 'trout','dolphin'],
                'animals':	['beaver', 'otter', 'seal', 'bear', 'leopard', 'lion', 'tiger', 'wolf','fox', 'porcupine', 'possum', 'raccoon',
                            'skunk','camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',\
                            'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                'people':['baby', 'boy', 'girl', 'man', 'woman']
                }



def parse_args():
    parser = argparse.ArgumentParser(description="Train FGVC Network")

    parser.add_argument(
        "--input_path",
        help="input train/test splitting files",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_path",
        help="save path for converted file ",
        type=str,
        required=False,
        default="."
    )
    args = parser.parse_args()
    return args

def read_and_decode(filename_queue):
    """Parses a single tf.Example into image and label tensors."""
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features["image"], tf.uint8)
    image.set_shape([3*32*32])
    label = tf.cast(features["label"], tf.int32)
    return image, label


def convert_from_tfrecords(data_root, dir_name, num_class, mode, output_path, json_file_prefix, class_to_super , class_names, super_class_names):
    if mode == 'valid':
        tfrecord_path = os.path.join(data_root, dir_name, 'eval.tfrecords')
    else:
        tfrecord_path = os.path.join(data_root, dir_name, 'train.tfrecords')
    filename_queue = tf.train.string_input_producer([tfrecord_path], shuffle=False, num_epochs=1)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    image, label = read_and_decode(filename_queue)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    annotations = []
    try:
        step = 0
        while not coord.should_stop():
            images, labels = sess.run([image, label])
            images = cv2.cvtColor(images.reshape(3, 32, 32).transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            im_path = os.path.join(output_path, json_file_prefix, 'images', str(labels))
            if not os.path.exists(im_path):
                os.makedirs(im_path)
            save_path = os.path.join(im_path, '{}_{}.jpg'.format(mode, step))
            cv2.imwrite(save_path, images)

            cls_name = class_names[int(labels)]
            super_category_id = class_to_super[int(labels)]
            super_category_name = super_class_names[super_category_id]
            #category_id = hierarchical[super_category_name].index(cls_name)
            category_id = int(labels)
            annotations.append({'fpath': save_path, 'image_id': step, 'category_id':category_id,\
                                'super_category_id':super_category_id, 'category_name': cls_name,\
                                'super_category_name':super_category_name})
            step += 1
    except tf.errors.OutOfRangeError:
        print('done')
    finally:
        coord.request_stop()

    with open(os.path.join(output_path, json_file_prefix, json_file_prefix+'_{}.json'.format(mode)), 'w') as f:
        json.dump({'annotations': annotations, 'num_classes': num_class, 'num_super_classes': len(super_class_names), 'num_classes_per_super_classes':[len(v) for k,v in hierarchical.items()]}, f)

    print('Json has been saved to', os.path.join(output_path, json_file_prefix, json_file_prefix+'_{}.json'.format(mode)))

if __name__ == '__main__':
    modes = ['train', 'valid']
    args = parse_args()

    cifar100_im50 = {'dir': 'cifar-100-data-im-0.02', 'json': 'cifar100_imbalance50', 'class':100}
    cifar100_im100 = {'dir': 'cifar-100-data-im-0.01', 'json': 'cifar100_imbalance100', 'class': 100}

    class_names_path = '/home/jrcai/hd/OLTR/CIFAR/converted/cifar100_clsname.json'
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    #class_names_path = '/home/jrcai/hd/OLTR/CIFAR/converted/cifar100_super_clsname.json'
    #with open(class_names_path, 'r') as f:
    #    super_class_names = json.load(f)
    super_class_names = list(hierarchical.keys())

    class_to_super = {}
    for super, sub_list in hierarchical.items():
        for sub_class in sub_list:
            class_to_super[class_names.index(sub_class)] = super_class_names.index(super)

    for m in modes:
        convert_from_tfrecords(
            args.input_path, cifar100_im100['dir'],
            cifar100_im100['class'], m, args.output_path,
            cifar100_im100['json'],
            class_to_super,
            class_names,
            super_class_names
        )
        convert_from_tfrecords(
            args.input_path, cifar100_im50['dir'],
            cifar100_im50['class'], m, args.output_path,
            cifar100_im50['json'],
            class_to_super,
            class_names,
            super_class_names
        )


