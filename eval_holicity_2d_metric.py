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


def Holicity_2d_metric(
        pre_seglist, pre_scorelist,
        annType="segm",
        output="data",
        annFile="HoliCity_valid_coco_format.json",  # gt json file name
        resFile='HoliCity_valid_results.json',  # results file name

):
    """
    all img_dirs, gt_seglist, pre_seglist, pre_scorelist, should be the list of absolute paths
    """

    if annType not in ['segm', 'bbox', 'keypoints']:
        raise ValueError("no such type!")

    # initialize COCO format ground truth api
    os.makedirs(output, exist_ok=True)
    annPath = osp.join(output, annFile)
    cocoGt = COCO(annPath)

    # initialize COCO results api
    resPath = osp.join(output, resFile)
    if not os.path.isfile(resPath):
        # the order of filelist should be same as gt
        coco_result_format(pre_seglist, pre_scorelist, resPath)
    cocoDt = cocoGt.loadRes(resPath)

    # running evaluation
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()  # will print the all results

    mAP = cocoEval.stats[0]


def V2_filelist(image_dirs=None):

    middle_p = sorted(os.listdir(image_dirs))
    filelist = []
    for pth in middle_p:
        imgs = sorted(os.listdir(osp.join(image_dirs, pth)))
        for img in imgs:
            filelist.append(osp.join(pth, img))
    return filelist


def eval_scannet():

    # gt_root = "/home/dxl/Data/LondonCity/renderings_plane/20200222"  # gt dirs
    # results_root = "logs/try/npz"  # results dir, should be like gt dirs
    # cleanf = "newfilelist.txt"
    # filef = "20200222-filelist.txt"
    # trainf = "20200222-middle-train.txt"
    # validf = "20200222-middle-test.txt"
    # filelist = get_filelist(rootdir=gt_root, split="valid", cleanf=cleanf, filef=filef, trainf=trainf, validf=validf)
    # img_dirs = [f"{gt_root.replace('renderings_plane', 'renderings')}/{path}_imag.png" for path in filelist]

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