import os
import os.path as osp
import copy
import numpy as np
import json
import glob
import time
import random
import datetime
from fvcore.common.file_io import PathManager

import torch
from tqdm import tqdm
import cv2
from pycocotools import mask
from HoliCity import Config
from HoliCity.utils import V1_filelist


_H, _W = 512, 512
_plane_area = 32 * 32


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, np.uint32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def toBbox(mask_array):
    idx = np.argwhere(mask_array == 1)
    # XYXY for detectron2 default format
    # return [np.min(idx[:, 1]), np.min(idx[:, 0]), np.max(idx[:, 1]), np.max(idx[:, 0])]

    # XYWH for COCO format
    return [
        np.min(idx[:, 1]),  # X
        np.min(idx[:, 0]),  # Y
        np.max(idx[:, 1]) - np.min(idx[:, 1]),  # W
        np.max(idx[:, 0]) - np.min(idx[:, 0])   # H
    ]


def COCO_format(img_dirs, gt_seg_dirs, output_folder, dataset_name):
    """
    :param img_dirs: list of absolute paths
    :param gt_seg_dirs: list of absolute paths
    :param output_folder:
    :param dataset_name:
    :return:
    """
    assert len(img_dirs) == len(gt_seg_dirs)
    categories = [
        {"id": 0, "name": "P"}
    ]

    coco_images = []
    coco_annotations = []
    for i, (im_path, gt_seg_path) in tqdm(enumerate(zip(img_dirs, gt_seg_dirs))):

        coco_image = {
            "id": i,
            "width": _W,
            "height": _H,
            "file_name": im_path,
        }
        coco_images.append(coco_image)
        idmap_face = cv2.imread(f"{gt_seg_path}", cv2.IMREAD_ANYDEPTH)
        unval = np.unique(idmap_face)

        area = []
        for val in unval[1:]:
            gt_mask = np.asarray(idmap_face == val, order="F")
            encoded_gt = mask.encode(gt_mask)
            area_gt = mask.area(encoded_gt)
            area.append(area_gt)
        # area = sorted(area)

        # top5 = area[-5] if len(area) > 6 else 0
        # print(sorted(area))
        # exit()

        for val in unval[1:]:
            coco_annotation = {}
            gt_mask = np.asarray(idmap_face == val, order="F")
            encoded_gt = mask.encode(gt_mask)

            area_gt = mask.area(encoded_gt)
            bbox_gt = toBbox(gt_mask)
            if area_gt < _plane_area:
                continue

            # if area_gt < top5:
            #     continue

            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox_gt]
            coco_annotation["segmentation"] = encoded_gt
            coco_annotation["area"] = area_gt
            coco_annotation["category_id"] = 0
            coco_annotation["iscrowd"] = 0

            coco_annotations.append(coco_annotation)

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "licenses": None,
    }
    print(
        "Conversion finished, "
        f"num images: {len(coco_images)}, num annotations: {len(coco_annotations)}"
    )

    cache_path = os.path.join(output_folder, f"{dataset_name}_coco_format.json")

    with PathManager.open(cache_path, "w") as json_file:
        # logger.info(f"Caching annotations in COCO format: {cache_path}")
        json.dump(coco_dict, json_file, cls=MyEncoder)

    return cache_path


def build_holicityV1_coco_format(name='HoliCityV1_train'):

    # configuration
    # -------- change with your path ----------
    cfg = Config()
    if cfg.machine == "dxl.cluster" and cfg.split_version != 'v1':
        raise ValueError()

    gt_image = f"{cfg.root}/image"
    gt_label = f"{cfg.root}/plane"
    json_out = cfg.json_out
    split_version = cfg.split_version  # v1, v1.1, v1.2
    # ------------------------------------------

    os.makedirs(json_out, exist_ok=True)

    filelist = V1_filelist(
        split=name.split("_")[1], split_version=split_version,
        rootdir=cfg.root
    )
    # print(filelist)
    # print(len(filelist))
    # exit()

    img_dirs = [osp.join(gt_image, path+"_imag.jpg") for path in filelist]
    gt_seglist = [osp.join(gt_label, path+"_plan.png") for path in filelist]

    cache_pth = COCO_format(img_dirs, gt_seglist, json_out, name)


def build_holicityV1_train():
    build_holicityV1_coco_format('HoliCityV1_train')


def build_holicityV1_valid():
    build_holicityV1_coco_format('HoliCityV1_valid')


def build_holicityV1_test():
    build_holicityV1_coco_format('HoliCityV1_test')


def build_holicityV1_test_valid():
    build_holicityV1_coco_format('HoliCityV1_test+valid')


def build_holicityV1_validhd():
    build_holicityV1_coco_format('HoliCityV1_validhd')
