import tensorflow as tf
import numpy as np
import os
import argparse
from tqdm import tqdm

from RecordReaderAll import *

# os.environ['CUDA_VISIBLE_DEVICES']=''

parser = argparse.ArgumentParser()
parser.add_argument('--input_tfrecords_file', type=str,
                    help='path .tfrecords file',
                    required=True)
parser.add_argument('--output_dir', type=str,
                    help='where to store extracted frames',
                    required=True)
parser.add_argument('--data_type', type=str,
                    help='where to store extracted frames',
                    required=True)
args = parser.parse_args()

input_tfrecords_file = args.input_tfrecords_file
output_dir = args.output_dir
data_type = args.data_type

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if data_type == 'train':
    file_list = open(output_dir + '/train.txt', 'w')
    output_dir = os.path.join(output_dir, 'train')
    os.makedirs(output_dir)
    max_num = 50000
elif data_type == 'val':
    file_list = open(output_dir + '/val.txt', 'w')
    output_dir = os.path.join(output_dir, 'val')
    os.makedirs(output_dir)
    max_num = 760
else:
    print("unsupported data type")
    exit(-1)


reader_train = RecordReaderAll()
filename_queue_train = tf.train.string_input_producer([input_tfrecords_file], num_epochs=1)
img_inp_train, global_gt_dict_train, local_gt_dict_train = reader_train.getBatch(filename_queue_train, batchSize=1, getLocal=True)

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in tqdm(range(max_num)):
        img, gt_dict = sess.run([img_inp_train, global_gt_dict_train])
        plane = gt_dict['plane'][0]  # [20, 3] the normal of each plane, padding to 20
        depth = gt_dict['depth'][0]  # [H, W, 1] the depth of each pixel
        normal = gt_dict['normal'][0]  # [H, W, 3]  all zero in valid
        semantics = gt_dict['semantics'][0]  # [H, W]
        segmentation = gt_dict['segmentation'][0]  # [H, W, 1]
        boundary = gt_dict['boundary'][0]   # [H, W, 2]
        num_planes = gt_dict['num_planes'][0].reshape([-1])  # [1, ] the number of planes
        image_path = gt_dict['image_path'][0]  # str: path of image
        info = gt_dict['info'][0]  # [20,] some information about each plane

        np.savez(os.path.join(output_dir, '%d.npz' % (i, )),
                 image=img[0], plane=plane, depth=depth, normal=normal, semantics=semantics,
                 segmentation=segmentation, boundary=boundary, num_planes=num_planes,
                 image_path=image_path, info=info)

        file_list.write('%d.npz\n' % (i, ))

        # if i % 1000 == 2:
        #     print(i)
        #     break

file_list.close()
