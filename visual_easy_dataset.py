import os
import copy
import numpy as np
import json
import glob
import time
import random
import datetime
from fvcore.common.file_io import PathManager

from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.layers import ROIAlign


import cv2
from pycocotools import mask
import matplotlib.pyplot as plt


GT = "/home/dxl/Data/r2/*[0-9]/*[0-9]_0_GL.npz"
# GT = r"D:\Download\cluster.Interface\stereoCAD\sample\*[0-9]_0_GL.npz"
_name = ""
_H, _W = 512, 512


def downsample_nearest(arr, new_size=(128, 128)):
    (h, w) = arr.shape
    x = np.arange(0, h, int(h/new_size[0]))
    y = np.arange(0, w, int(w/new_size[1]))
    assert len(x) == new_size[0] and len(y) == new_size[1]
    x_, y_ = np.meshgrid(x, y)

    new_arr = arr[y_.flatten(), x_.flatten()]
    return new_arr.reshape(new_size)


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


def COCO_format(dirs, output_folder, dataset_name):
    # dirs = sorted(glob.glob(GT))[:_num]
    categories = [
        {"id": 0, "name": "P"}
    ]

    coco_images = []
    coco_annotations = []
    t = time.time()
    for i, path in enumerate(dirs):
        if i % 1000 == 0:
            print(
                f"num images: {i}, times: {time.time() - t:.1f}"
            )
        coco_image = {
            "id": i,
            "width": _W,
            "height": _H,
            "file_name": path.replace("_GL.npz", ".png"),
        }
        coco_images.append(coco_image)

        with open(path.replace("_GL.npz", "_label.json"), "r") as f:
            data = json.load(f)
            tris2plane = np.array(data["tris2plane"])
        with np.load(path) as gl:
            idmap_face = gl["idmap_face"]
            idmap_face = np.array(tris2plane[idmap_face.ravel() - 1]).reshape(idmap_face.shape)
            idmap_face = downsample_nearest(idmap_face, new_size=(_H, _W))

        unval, idx = np.unique(idmap_face, return_inverse=True)

        # the background id from 0 to max after tris2plane !!!
        for val in unval[:-1]:
            coco_annotation = {}
            gt_mask = np.asarray(idmap_face == val, order="F")
            encoded_gt = mask.encode(gt_mask)

            area_gt = mask.area(encoded_gt)
            # print(area_gt)
            # print(type(area_gt))
            # print(str(area_gt))
            # exit()
            bbox_gt = toBbox(gt_mask)
            if area_gt < 10:
                continue

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


def get_planar_dicts(num):
    dirs = sorted(glob.glob(GT))[:num]

    dataset_dicts = []
    for i, path in enumerate(dirs):
        record = {}
        record["file_name"] = path.replace("_GL.npz", ".png")
        record["image_id"] = i
        record["height"] = _H
        record["width"] = _W

        # with open(path.replace("_GL.npz", "_label.json"), "r") as f:
        #     data = json.load(f)
        #     tris2plane = np.array(data["tris2plane"])
        with np.load(path) as gl:
            idmap_face = gl["idmap_face"]

            # idmap_face = np.array(tris2plane[idmap_face.ravel() - 1]).reshape(idmap_face.shape)

            idmap_face = downsample_nearest(idmap_face, new_size=(_H, _W))

        objs = []

        unval, idx = np.unique(idmap_face, return_inverse=True)
        # print(unval)
        # exit()

        # the background id from 0 to max after tris2plane !!!
        for val in unval[:-1]:
            # if val == 0:
            #     continue
            gt_mask = np.asarray(idmap_face == val, order="F")
            encoded_gt = mask.encode(gt_mask)
            area_gt = mask.area(encoded_gt)
            bbox_gt = mask.toBbox(encoded_gt)
            # print(area_gt)
            # exit()

            obj = {
                # "bbox": bbox_gt.tolist(),
                "bbox": toBbox(gt_mask),
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": encoded_gt,
                "area": area_gt,
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def make_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.DATASETS.TRAIN = (_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0149999.pth")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 150000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (plane)

    # user parameters
    cfg.OUTPUT_DIR = "./output/synthesized_output"
    cfg.INPUT.MASK_FORMAT = "bitmask"  # polygon

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def evaluator(cfg):
    evaluator = COCOEvaluator(_name, cfg, False, output_dir="./output/synthesized_output")
    val_loader = build_detection_test_loader(cfg, _name)

    return evaluator, val_loader


def build_predictor_vis(cfg):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0149999.pth")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg.DATASETS.TEST = (_name,)
    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get(_name)
    metadata = MetadataCatalog.get(_name)
    for k in range(100):
        d = dataset_dicts[k]
    # for d in random.sample(dataset_dicts, 2):
        im = cv2.imread(d["file_name"])
        im_demo = cv2.imread(d["file_name"].replace(".png", "_demo.jpg"))
        outputs = predictor(im)
        v_p = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=2,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        v_gt = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=2,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        v_p_nobbx = Visualizer(im_demo[:, :, ::-1],
                       metadata=metadata,
                       scale=2,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )

        v_p = v_p.draw_instance_predictions(outputs["instances"].to("cpu"))
        v_gt = v_gt.draw_dataset_dict(d)
        outputs["instances"].pred_boxes.tensor[:] = 0
        v_p_nobbx = v_p_nobbx.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.figure(figsize=[40, 10])
        plt.subplot(141), plt.imshow(im_demo)
        plt.subplot(142), plt.imshow(v_p_nobbx.get_image()[:, :, ::-1])
        plt.subplot(143), plt.imshow(v_p.get_image()[:, :, ::-1])
        plt.subplot(144), plt.imshow(v_gt.get_image()[:, :, ::-1])
        plt.savefig(f"demo/synthesized_easy/predict_{d['image_id']}.png", dpi=200), plt.close()
        print(f"Saving demo/synthesized_easy/predict_{d['image_id']}.png")


def random_vis():
    dataset_dicts = DatasetCatalog.get(_name)
    metadata = MetadataCatalog.get(_name)
    # dataset_dicts = get_planar_dicts()
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        im_demo = cv2.imread(d["file_name"].replace(".png", "_demo.jpg"))
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)

        plt.figure(figsize=[20, 10])
        plt.subplot(121), plt.imshow(vis.get_image()[:, :, ::-1])
        plt.subplot(122), plt.imshow(im_demo)
        plt.savefig(f"demo/visual_{d['image_id']}.png"), plt.close()
        print(f"Saving demo/visual_{d['image_id']}.png")


def assign_global(name):
    global _name
    _name = name


if __name__ == '__main__':
    # _num = 600
    _is_train = False
    assign_global("synthesized_easy_val")
    # dirs = sorted(glob.glob(GT))[:_num]
    # cache_pth = COCO_format(dirs, "data/", _name)
    # exit()
    cache_pth = f"data/{_name}_coco_format.json"
    register_coco_instances(name=_name,
                            metadata={'thing_classes': ["P"]},
                            json_file=cache_pth,
                            image_root="/home/dxl/Data/r2/")

    # random visualization
    # random_vis()
    # exit()

    cfg__ = make_cfg()

    # begin train
    trainer = DefaultTrainer(cfg__)
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    # visualizing validation sample
    build_predictor_vis(cfg__)

    # evaluating validation data set
    trainer.resume_or_load(resume=True)
    evaluator, val_loader = evaluator(cfg__)
    inference_on_dataset(trainer.model, val_loader, evaluator)


