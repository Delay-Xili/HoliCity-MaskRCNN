"""Evaluate sAP5, sAP10, sAP15 for LCNN
Usage:
    eval-sAP.py [options]
    eval-sAP.py (-h | --help )

Arguments:
    <path>                           One or more directories from train.py

Options:
   -h --help                         Show this screen.
   -d --device <device>              device config   [default: dxl.cluster]
   -s --stage <stage>              stage           [default: train]
"""

import os
import os.path as osp
import numpy as np
from docopt import docopt

import torch
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import cv2

from ScanNet.Coco_format import convert_planenet2coco


_name = ""



class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # evaluator_list = []

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def make_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.DATASETS.TRAIN = (_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0099999.pth")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 150000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (plane)

    # user parameters
    cfg.OUTPUT_DIR = "./output/holiCity_output"
    cfg.INPUT.MASK_FORMAT = "bitmask"  # polygon

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def visualize_data(output):
    dataset_dicts = DatasetCatalog.get(_name)
    metadata = MetadataCatalog.get(_name)
    output = f"{output}/viz"
    os.makedirs(output, exist_ok=True)

    for k in range(5):
        d = dataset_dicts[k]
        im = cv2.imread(d["file_name"])
        print(d["file_name"])
        # print(type(im))
        v_gt = Visualizer(
            im[:, :, ::-1], metadata=metadata, scale=2,
            instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
         )

        v_gt = v_gt.draw_dataset_dict(d)
        cv2.imwrite(f"{output}/viz_{d['image_id']}_gt.png", v_gt.get_image())
        print(f"saving {output}/viz_{d['image_id']}_gt.png")


def save_npz(cfg, dataset_dicts, metadata, isnpz=True, isviz=True):

    predictor = DefaultPredictor(cfg)

    for k in tqdm(range(len(dataset_dicts))):
        d = dataset_dicts[k]
        # print(d["file_name"])
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        if isviz:
            v_p = Visualizer(im[:, :, ::-1],
                             metadata=metadata,
                             scale=2,
                             instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                             )

            v_p = v_p.draw_instance_predictions(outputs["instances"].to("cpu"))
            outputs["instances"].pred_boxes.tensor[:] = 0

            image_dir = osp.dirname(d["file_name"])
            mid_dir = osp.basename(image_dir)
            prefix = f"{cfg.OUTPUT_DIR}/viz/{mid_dir}"
            os.makedirs(prefix, exist_ok=True)

            cv2.imwrite(f"{prefix}/{osp.basename(d['file_name'])}", v_p.get_image())

        if isnpz:
            masks = outputs["instances"].pred_masks
            scores = outputs["instances"].scores

            if masks.shape[0] != scores.shape[0]:
                raise ValueError("ggg")

            segmentation = torch.zeros_like(masks).sum(0)

            for i in range(masks.shape[0]):
                segmentation[masks[i]] = i + 1

            image_name = osp.basename(d["file_name"])
            image_name = image_name[:-4]

            # image_dir = osp.dirname(d["file_name"])
            # print(d["file_name"])
            # exit()
            # mid_dir = osp.basename(image_dir)
            prefix = f"{cfg.OUTPUT_DIR}/npz"
            os.makedirs(prefix, exist_ok=True)

            prefix = f"{prefix}/{image_name}"
            cv2.imwrite(f"{prefix}_plan.png", segmentation.cpu().numpy().astype(np.uint16))
            np.savez(f"{prefix}_plan.npz", scores=scores.cpu().numpy())


def assign_global(name):
    global _name
    _name = name


def train():

    # isws399 = False
    # if not isws399:
    #     raise ValueError("warning: only support the ws399 machine, not dxl.cluster!")

    # configuration
    name = "ScanNet_train"  # "ScanNet_train" "ScanNet_val"
    assign_global(name)
    json_out = "data"
    image_root = "./"
    output_dir = f"./output/{_name}_output"
    # ------------------------------------------

    # ----step 1------
    # data prepare
    cache_pth = f"{json_out}/{_name}_coco_format.json"
    if not os.path.isfile(cache_pth):

        split = name.split('_')[1]
        root_npz = f"dataset/scannet/{split}"
        splitxt = f"dataset/scannet/{split}.txt"
        output_img_folder = f"dataset/scannet/{split}_image"

        cache_pth = convert_planenet2coco(
            root_npz, splitxt, output_img_folder, output_folder=json_out, dataset_name=f"{_name}",
        )
        exit()

    register_coco_instances(name=_name,
                            metadata={'thing_classes': ["P"]},
                            json_file=cache_pth,
                            image_root=image_root,
                            )

    # visualize_data(output_dir+"/viz_train_data")
    # exit()

    # begin train
    cfg__ = make_cfg()

    # -----add eval--
    register_coco_instances(name="ScanNet_val",
                            metadata={'thing_classes': ["P"]},
                            json_file=f"{json_out}/ScanNet_val_coco_format.json",
                            image_root=image_root,
                            )
    cfg__.DATASETS.TEST = ("ScanNet_val", )
    cfg__.TEST.EVAL_PERIOD = 5000
    cfg__.OUTPUT_DIR = output_dir
    trainer = Trainer(cfg__)
    # ---------------

    # cfg__.OUTPUT_DIR = output_dir
    # trainer = DefaultTrainer(cfg__)

    trainer.resume_or_load(resume=True)
    trainer.train()


def valid():

    from eval_2d_metric import Holicity_2d_metric

    # isws399 = False
    # if not isws399:
    #     raise ValueError("warning: only support the ws399 machine, not dxl.cluster!")

    # configuration
    name = "ScanNet_val"  # "ScanNet_train" "ScanNet_val"
    assign_global(name)
    json_out = "data"
    image_root = "./"
    output_dir = f"./output/{_name}_output"
    ckpt = os.path.join("./output/ScanNet_train_output", "model_0099999.pth")
    # ------------------------------------------

    # ----step 1------
    # data prepare
    split = name.split('_')[1]
    root_npz = f"dataset/scannet/{split}"
    splitxt = f"dataset/scannet/{split}.txt"
    output_img_folder = f"dataset/scannet/{split}_image"

    cache_pth = f"{json_out}/{_name}_coco_format.json"
    if not os.path.isfile(cache_pth):

        cache_pth = convert_planenet2coco(
            root_npz, splitxt, output_img_folder, output_folder=json_out, dataset_name=f"{_name}",
        )
        exit()

    register_coco_instances(name=_name,
                            metadata={'thing_classes': ["P"]},
                            json_file=cache_pth,
                            image_root=image_root,
                            )

    cfg__ = make_cfg()
    cfg__.OUTPUT_DIR = output_dir

    # evaluating validation data set
    cfg__.MODEL.WEIGHTS = ckpt
    cfg__.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg__.DATASETS.TEST = (_name,)
    metadata = MetadataCatalog.get(_name)
    dataset_dicts = DatasetCatalog.get(_name)

    save_npz(cfg__, dataset_dicts=dataset_dicts, metadata=metadata, isnpz=True, isviz=False)

    # eval 2d metric
    filelist = sorted(os.listdir(output_img_folder))
    results_root = cfg__.OUTPUT_DIR+"/npz"
    Holicity_2d_metric(
        img_dirs=None, gt_seglist=None,
        pre_seglist=[osp.join(results_root, path.replace(".jpg", "_plan.png")) for path in filelist],
        pre_scorelist=[osp.join(results_root, path.replace(".jpg", "_plan.npz")) for path in filelist],
    )


def predict():
    # isws399 = False
    # if not isws399:
    #     raise ValueError("warning: only support the ws399 machine, not dxl.cluster!")

    # TODO configuration
    name = ""  # "customer dataset name"
    assign_global(name)
    image_root = ""
    output_dir = f"./output/{_name}_output"
    ckpt = os.path.join("./output/ScanNet_train_output", "model_0099999.pth")
    # ------------------------------------------

    MetadataCatalog.get(name).set(
        evaluator_type="coco", **{'thing_classes': ["P"]}
    )

    cfg__ = make_cfg()
    cfg__.OUTPUT_DIR = output_dir

    # evaluating validation data set
    cfg__.MODEL.WEIGHTS = ckpt
    cfg__.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg__.DATASETS.TEST = (_name,)
    metadata = MetadataCatalog.get(_name)

    filelist = sorted(os.listdir(image_root))
    dataset_dicts = [{"file_name": p} for p in filelist]
    isnpz = True
    isviz = True

    save_npz(cfg__, dataset_dicts=dataset_dicts, metadata=metadata, isnpz=isnpz, isviz=isviz)


if __name__ == '__main__':
    args = docopt(__doc__)

    if args["--device"] != "w399":
        raise ValueError("warning: only support the ws399 machine, not dxl.cluster!")

    if args["--stage"] == "train":
        train()
    elif args["--stage"] == "valid":
        valid()
    elif args["--stage"] == "predict":
        predict()
    else:
        raise ValueError("")

