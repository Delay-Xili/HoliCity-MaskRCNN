"""Evaluate sAP5, sAP10, sAP15 for LCNN
Usage:
    main.py [options]
    main.py (-h | --help )

Examples:
    python main.py -s train -m HoliCityV1

Arguments:
    <path>                           One or more directories from train.py

Options:
   -h --help                         Show this screen.
   -s --stage <stage>                stage           [default: train]
   -m --mode <mode>                  dataset         [default: HoliCityV1]
"""

import os
import os.path as osp
import numpy as np
import itertools
import json
from docopt import docopt
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from eval_2d_metric import MyEncoder

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
from detectron2.structures import Boxes, BoxMode, pairwise_iou

import cv2
from tools import yichao_visual

H, W = 512, 512


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
    # cfg.DATASETS.TRAIN = (name,)
    # cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0099999.pth")
    cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # cfg.SOLVER.STEPS = (30000,)
    cfg.SOLVER.MAX_ITER = 150000  #

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (plane)

    # user parameters
    # cfg.OUTPUT_DIR = "./output/holiCity_output"
    cfg.INPUT.MASK_FORMAT = "bitmask"  # polygon

    return cfg


def inference(cfg, predictor, dataset_dicts, isnpz=True, isviz=True, meta_dir_mode="1"):

    os.makedirs(f"{cfg.OUTPUT_DIR}/viz", exist_ok=True)
    os.makedirs(f"{cfg.OUTPUT_DIR}/npz", exist_ok=True)

    for k in tqdm(range(len(dataset_dicts))):
        d = dataset_dicts[k]
        im = cv2.imread(d["file_name"])  # the output of cv2.imread is follow BGR channel
        im = cv2.resize(im, (H, W))
        outputs = predictor(im)
        masks = outputs["instances"].pred_masks
        scores = outputs["instances"].scores

        if isviz:
            if meta_dir_mode == "2":
                image_dir = osp.dirname(d["file_name"])
                mid_dir = osp.basename(image_dir)
                prefix = f"{cfg.OUTPUT_DIR}/viz/{mid_dir}"
                os.makedirs(prefix, exist_ok=True)
                prefix = f"{prefix}/{osp.basename(d['file_name'])}"
            elif meta_dir_mode == "1":
                prefix = f"{cfg.OUTPUT_DIR}/viz/{osp.basename(d['file_name'])}"
            else:
                raise ValueError()

            masks = outputs["instances"].pred_masks
            masks = masks.permute(1, 2, 0).cpu().numpy()
            yichao_visual(im[:, :, ::-1] / 255.0, masks, prefix)

        if isnpz:

            if masks.shape[0] != scores.shape[0]:
                raise ValueError("ggg")

            segmentation = torch.zeros_like(masks).sum(0)

            for i in range(masks.shape[0]):
                # since the segmentation results of maskrcnn has overlap among predicted segmentation,
                # so we merge all prediction into on 'png' format with score increased order,
                # make the mask with low score be covered by mask with high score
                l = masks.shape[0] - 1 - i
                segmentation[masks[l]] = l + 1

            if meta_dir_mode == "2":
                # holicity data storage style (2 level storage format)
                image_dir = osp.dirname(d["file_name"])
                mid_dir = osp.basename(image_dir)
                prefix = f"{cfg.OUTPUT_DIR}/npz/{mid_dir}"
                os.makedirs(prefix, exist_ok=True)
                # TODO
                prefix = f"{prefix}/{osp.basename(d['file_name'])[:-9]}"
            elif meta_dir_mode == "1":
                # planenet data storage style (1 level storage format)
                # TODO
                prefix = f"{cfg.OUTPUT_DIR}/npz/{osp.basename(d['file_name'])[:-4]}"
            else:
                raise ValueError()

            cv2.imwrite(f"{prefix}_plan.png", segmentation.cpu().numpy().astype(np.uint16))
            np.savez(f"{prefix}_plan.npz", scores=scores.cpu().numpy())


def train(args):

    # configuration
    register_coco_instances(name=args.train_name,
                            metadata=args.metadata,
                            json_file=f"{args.json_out}/{args.train_json_file}",
                            image_root=args.image_root,
                            )

    # begin train
    cfg__ = make_cfg()
    cfg__.DATASETS.TRAIN = (args.train_name,)
    cfg__.SOLVER.BASE_LR = args.lr
    cfg__.SOLVER.STEPS = (args.lr_decay,)

    # -----add eval--
    register_coco_instances(name=args.val_name,
                            metadata=args.metadata,
                            json_file=f"{args.json_out}/{args.val_json_file}",
                            image_root=args.image_root,
                            )
    cfg__.DATASETS.TEST = (args.val_name, )
    cfg__.TEST.EVAL_PERIOD = args.eval_period
    cfg__.OUTPUT_DIR = args.train_output_dir
    os.makedirs(cfg__.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg__)
    # ---------------

    trainer.resume_or_load(resume=True)
    trainer.train()


def validation(
        name, metadata, json_file, image_root, output_dir,
        ckpt, score_thresh_test, output_meta_dirs, meta_dir_mode,
):

    register_coco_instances(name=name,
                            metadata=metadata,
                            json_file=json_file,
                            image_root=image_root,
                            )

    cfg__ = make_cfg()
    cfg__.OUTPUT_DIR = output_dir
    os.makedirs(cfg__.OUTPUT_DIR, exist_ok=True)

    # evaluating validation data set
    cfg__.MODEL.WEIGHTS = ckpt
    cfg__.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test  # set the testing threshold for this model
    cfg__.DATASETS.TEST = (name,)
    predictor = DefaultPredictor(cfg__)

    evaluator = COCOEvaluator(name, cfg__, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg__, name)
    inference_on_dataset(predictor.model, val_loader, evaluator)

    # eval 2d metric
    # use_our_2d_metric = False
    # if use_our_2d_metric:
    #
    #     from eval_2d_metric import coco_2d_metric, build_holicity_result_json
    #
    #     dataset_dicts = DatasetCatalog.get(name)
    #
    #     inference(
    #         cfg__, predictor,
    #         dataset_dicts=dataset_dicts, isnpz=True, isviz=False, meta_dir_mode=meta_dir_mode
    #     )
    #
    #     build_holicity_result_json(results_root=cfg__.OUTPUT_DIR, image_root=image_root)
    #
    #     coco_2d_metric(
    #         annType="segm",
    #         annFile=json_file,
    #         resFile=f"{cfg__.OUTPUT_DIR}/result_json/coco_results.json",
    #     )
    #
    # else:
    #     # prefer this metric
    #     # no meta results, measure the results on-line
    #     evaluator = COCOEvaluator(name, cfg__, False, output_dir=output_dir)
    #     val_loader = build_detection_test_loader(cfg__, name)
    #     inference_on_dataset(predictor.model, val_loader, evaluator)


def predict(args):

    cfg__ = make_cfg()
    cfg__.OUTPUT_DIR = args.pred_output_dir
    os.makedirs(cfg__.OUTPUT_DIR, exist_ok=True)

    # evaluating validation data set
    cfg__.MODEL.WEIGHTS = args.ckpt
    cfg__.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh_test  # set the testing threshold for this model
    predictor = DefaultPredictor(cfg__)

    dataset_dicts = [{"file_name": p} for p in args.img_dirs_list]
    isnpz = False
    isviz = True

    inference(
        cfg__, predictor,
        dataset_dicts=dataset_dicts, isnpz=isnpz, isviz=isviz, meta_dir_mode=args.meta_dir_mode
    )


if __name__ == '__main__':

    args = docopt(__doc__)

    # ---------- choose dataset -------------
    if args["--mode"] == "HoliCityV1":
        from HoliCity import Config
        from HoliCity.Coco_format import build_holicityV1_coco_format as build_coco_format
    elif args["--mode"] == "ScanNet":
        from ScanNet import Config
        from ScanNet.Coco_format import build_scannet_coco_format as build_coco_format
    elif args["--mode"] == "Megadepth":
        from Megadepth import Config
    elif args["--mode"] == "SYNTHIA":
        from SYNTHIA import Config
    else:
        raise ValueError()

    config = Config()

    # --------- choose split ------------------
    if args["--stage"] == "train":
        config.training()
        if not os.path.isfile(f"{config.json_out}/{config.train_json_file}"):
            build_coco_format(f"{args['--mode']}_train")

        config.valid(postfix="valid")
        if not os.path.isfile(f"{config.json_out}/{config.val_json_file}"):
            build_coco_format(f"{args['--mode']}_valid")

        train(config)
    elif args["--stage"] in ["valid", 'test', 'test+valid', 'validhd', 'validld']:
        assert config.ckpt is not None
        print("pretrained ckpt: ", config.ckpt)
        config.valid(postfix=args["--stage"])
        if args["--stage"] in ['test', 'test+valid', 'validhd', 'validld']:
            assert args["--mode"] == "HoliCityV1"
        if not os.path.isfile(f"{config.json_out}/{config.val_json_file}"):
            build_coco_format(f"{args['--mode']}_{args['--stage']}")
        validation(
            name=config.val_name, metadata=config.metadata,
            json_file=f"{config.json_out}/{config.val_json_file}",
            image_root=config.image_root, output_dir=config.val_output_dir, ckpt=config.ckpt,
            score_thresh_test=config.score_thresh_test, output_meta_dirs=config.valid_meta_dirs,
            meta_dir_mode=config.meta_dir_mode
        )
    elif args["--stage"] == "predict":
        assert config.ckpt is not None
        print("pretrained ckpt: ", config.ckpt)
        config.predict()
        predict(config)
    else:
        raise ValueError("")

