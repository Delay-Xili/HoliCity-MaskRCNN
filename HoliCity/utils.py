import os
import os.path as osp
import numpy as np

import torch
from tqdm import tqdm
import cv2

_H, _W = 512, 512
_plane_area = 32 * 32


def get_image_dir_list(image_dirs=None):

    middle_p = sorted(os.listdir(image_dirs))
    filelist = []
    for pth in middle_p:
        imgs = sorted(os.listdir(osp.join(image_dirs, pth)))
        for img in imgs:
            filelist.append(osp.join(pth, img))
    return filelist


def V1_filelist(split='train', rootdir=None, split_version='v1',
                cleanf="clean-filelist.txt",
                filef="filelist.txt",
                trainf="train-middlesplit.txt",
                validf="valid-middlesplit.txt",
                validhd="valid-hd.txt",
                testf="test-middlesplit.txt",
                ):

    split_dir = osp.join(rootdir, "split", split_version)
    image_dir = osp.join(rootdir, "image")
    plane_dir = osp.join(rootdir, "plane")
    if osp.exists(osp.join(split_dir, cleanf)):
        print(f"{osp.join(split_dir, cleanf)} already exited")
        with open(osp.join(split_dir, cleanf), "r") as f:
            samples = [line[:-1] for line in f.readlines()]
    else:
        with open(osp.join(split_dir, filef), "r") as f:
            samples = [line[:-1] for line in f.readlines()]
        samples = data_clean(plane_dir, samples, cleanf, split_dir)

    if split in ["train", "valid", "test"]:
        with open(osp.join(split_dir, eval(split+'f')), "r") as ft:
            check_sample = [line[:-1] for line in ft.readlines()]
    elif split == "test+valid":
        with open(osp.join(split_dir, validf), "r") as ft:
            check_sample_v = [line[:-1] for line in ft.readlines()]
        with open(osp.join(split_dir, testf), "r") as ft:
            check_sample_t = [line[:-1] for line in ft.readlines()]
        check_sample = check_sample_v + check_sample_t
    elif split == "validhd":
        with open(osp.join(split_dir, validhd), "r") as ft:
            check_sample = [line[:-1] for line in ft.readlines()]
    elif split == "validld":
        with open(osp.join(split_dir, validf), "r") as ft:
            check_sample = [line[:-1] for line in ft.readlines()]
        check_sample = [s for s in check_sample if s[-2:] == "LD"]
    else:
        raise ValueError("")

    filelist = []
    for file in samples:
        if file[:-7] in check_sample:
            filelist.append(file)

    return filelist


def data_clean(plane_dir, filelist, cleanf, output):

    new_file = []
    print("data cleaning......")
    for name_ in tqdm(filelist):
        name = osp.join(plane_dir, name_)
        # if os.path.isfile(name + "_plan.npz"):
        #     pass
        # else:
        #     print(name)
        # continue

        with np.load(name + "_plan.npz") as npz:
            # plane = data['plane']
            planes_ = npz['ws']

        gt_segmentation = cv2.imread(name + "_plan.png", cv2.IMREAD_ANYDEPTH)

        uni_idx = np.unique(gt_segmentation)
        planes_in_seg = len(uni_idx)
        planes_in_nor = planes_.shape[0]

        plane_area = []
        for idx in uni_idx[1:]:
            plane_area.append(np.sum(gt_segmentation == idx))  # (num_planes, )
        plane_area = np.array(plane_area)
        sat_plane_num = np.sum(plane_area > _plane_area)

        if planes_in_seg >= 2 and planes_in_nor + 1 >= planes_in_seg and sat_plane_num >=2:
            new_file.append(name_+"\n")

    with open(f"{output}/{cleanf}", "w") as f:
        f.writelines(new_file)
    print(f" The original files number: {len(filelist)}, and the cleaned files number: {len(new_file)}. {len(filelist)-len(new_file)} removed!")
    print(f"Saving {output}/{cleanf}")
    exit()

    return [line[:-1] for line in new_file]
