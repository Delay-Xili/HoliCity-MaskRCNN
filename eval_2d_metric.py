import os
import os.path as osp
import copy
import numpy as np
import json
import glob
import time
import random
import datetime
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
from fvcore.common.file_io import PathManager

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
import numpy as np

_H, _W = 512, 512
_plane_area = 32 * 32


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, np.uint32):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
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


def coco_result_format(pre_seg_dirs, pre_score_dirs, outpath):
    """
    the order should be same as gt
    pre_seg_dirs: list of absolute paths
    pre_score_dirs: list of absolute paths
    """
    coco_results = []

    for i, (seg_pth, score_pth) in enumerate(zip(pre_seg_dirs, pre_score_dirs)):
        idmap_face = cv2.imread(seg_pth, cv2.IMREAD_ANYDEPTH)
        unval = np.unique(idmap_face)

        with np.load(score_pth) as npz:
            s = npz["scores"]

        rles, scores = [], []
        for j, val in enumerate(unval[1:]):
            pmask = np.asarray(idmap_face == val, order="F")
            encoded_p = mask.encode(pmask)
            area = mask.area(encoded_p)

            if area > _plane_area:
                rles.append(encoded_p)
                scores.append(s[j])

        coco_results.extend(
            [
                {
                    "image_id": i,
                    "category_id": 0,
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    with PathManager.open(outpath, "w") as json_file:
        # logger.info(f"Caching annotations in COCO format: {cache_path}")
        json.dump(coco_results, json_file, cls=MyEncoder)


def coco_2d_metric(
        annType="segm",
        annFile="HoliCity_valid_coco_format.json",  # gt json file name
        resFile='HoliCity_valid_results.json',  # results file name
        ):

    if annType not in ['segm', 'bbox', 'keypoints']:
        raise ValueError("no such type!")

    # initialize COCO format ground truth api
    cocoGt = COCO(annFile)

    # initialize COCO results api
    cocoDt = cocoGt.loadRes(resFile)

    # running evaluation
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()  # will print the all results

    mAP = cocoEval.stats[0]


def build_holicity_result_json(results_root, image_root, split='valid', split_version='v1'):
    from HoliCity.utils import V1_filelist

    npz_pth = f"{results_root}/npz"
    json_pth = f"{results_root}/result_json"
    os.makedirs(json_pth, exist_ok=True)

    output = f"{json_pth}/coco_results.json"

    # image_root = "/home/dxl/Data/LondonCity/V1"
    meta_dirs = V1_filelist(split=split, rootdir=image_root, split_version=split_version)

    pre_seglist = [osp.join(npz_pth, path + "_plan.png") for path in meta_dirs]
    pre_scorelist = [osp.join(npz_pth, path + "_plan.npz") for path in meta_dirs]

    coco_result_format(pre_seglist, pre_scorelist, output)
    print("build result.json Done")


def eval_scannet():

    # configuration
    # -------- change with your path ----------
    gt_image = "dataset/scannet/val_image"
    gt_label = "dataset/scannet/val"
    name = "ScanNet"
    output = "data"
    # results_root = "/home/dxl/Code/try_detectron2/output/HoliCity_valid_scannet_pretrained_output/npz/"

    # output = "data/HoliCityV2_valid_AE_scannet_pretrained"
    results_root = "output/ScanNet_val_output/npz"
    # ------------------------------------------

    # filelist = V2_filelist(gt_image)
    filelist = sorted(os.listdir(gt_image))
    img_dirs = None
    gt_seglist = None
    pre_seglist = [osp.join(results_root, path.replace(".jpg", "_plan.png")) for path in filelist]
    pre_scorelist = [osp.join(results_root, path.replace(".jpg", "_plan.npz")) for path in filelist]

    Holicity_2d_metric(img_dirs, gt_seglist, pre_seglist, pre_scorelist,
                       annFile=f"{name}_val_coco_format.json",  # gt json file name
                       resFile=f"{name}_val_results.json",  # results file name
                       output=output
                       )


if __name__ == '__main__':
    eval_scannet()