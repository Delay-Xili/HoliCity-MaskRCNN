"""Evaluate sAP5, sAP10, sAP15 for LCNN
Usage:
    eval-sAP.py [options]
    eval-sAP.py (-h | --help )

Arguments:
    <path>                           One or more directories from train.py

Options:
   -h --help                         Show this screen.
   -d --device <device>              device config   [default: dxl.cluster]
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
from eval_holicity_2d_metric import MyEncoder

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


# def visualize_data(output, name):
#     dataset_dicts = DatasetCatalog.get(name)
#     metadata = MetadataCatalog.get(name)
#     output = f"{output}/viz"
#     os.makedirs(output, exist_ok=True)
#
#     for k in range(5):
#         d = dataset_dicts[k]
#         im = cv2.imread(d["file_name"])
#         print(d["file_name"])
#         # print(type(im))
#         v_gt = Visualizer(
#             im[:, :, ::-1], metadata=metadata, scale=2,
#             instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
#          )
#
#         v_gt = v_gt.draw_dataset_dict(d)
#         cv2.imwrite(f"{output}/viz_{d['image_id']}_gt.png", v_gt.get_image()[:, :, ::-1])
#         print(f"saving {output}/viz_{d['image_id']}_gt.png")


# def instances_to_coco_json(instances, img_id):
#     """
#     Dump an "Instances" object to a COCO-format json that's used for evaluation.
#
#     Args:
#         instances (Instances):
#         img_id (int): the image id
#
#     Returns:
#         list[dict]: list of json annotations in COCO format.
#     """
#     num_instance = len(instances)
#     if num_instance == 0:
#         return []
#
#     # boxes = instances.pred_boxes.tensor.numpy()
#     # boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
#     # boxes = boxes.tolist()
#     scores = instances.scores.tolist()
#     classes = instances.pred_classes.tolist()
#
#     # use RLE to encode the masks, because they are too large and takes memory
#     # since this evaluator stores outputs of the entire dataset
#     rles = [
#         mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
#         for mask in instances.pred_masks
#     ]
#     for rle in rles:
#         # "counts" is an array encoded by mask_util as a byte-stream. Python3's
#         # json writer which always produces strings cannot serialize a bytestream
#         # unless you decode it. Thankfully, utf-8 works out (which is also what
#         # the pycocotools/_mask.pyx does).
#         rle["counts"] = rle["counts"].decode("utf-8")
#
#     results = []
#     for k in range(num_instance):
#         result = {
#             "image_id": img_id,
#             "category_id": classes[k],
#             "segmentation": rles[k],
#             "score": scores[k],
#         }
#         results.append(result)
#         # print(results)
#         # exit()
#     return results
#
#
# def inference_(predictor, dataset_dicts):
#     predictions = []
#     for k in tqdm(range(len(dataset_dicts))):
#         d = dataset_dicts[k]
#         # print(d["file_name"])
#         im = cv2.imread(d["file_name"])
#         outputs = predictor(im)
#         predictions.append(outputs["instances"].to(torch.device("cpu")))
#
#     return predictions


def inference(cfg, predictor, dataset_dicts, metadata, isnpz=True, isviz=True, meta_dir_mode="1"):

    # predictor = DefaultPredictor(cfg)
    os.makedirs(f"{cfg.OUTPUT_DIR}/viz", exist_ok=True)
    os.makedirs(f"{cfg.OUTPUT_DIR}/npz", exist_ok=True)

    for k in tqdm(range(len(dataset_dicts))):
        d = dataset_dicts[k]
        # print(d["file_name"])
        im = cv2.imread(d["file_name"])  # the output of cv2.imread is follow BGR channel
        im = cv2.resize(im, (H, W))
        outputs = predictor(im)

        if isviz:
            # v_p = Visualizer(im[:, :, ::-1],
            #                  metadata=metadata,
            #                  scale=2,
            #                  instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
            #                  )
            #
            # v_p = v_p.draw_instance_predictions(outputs["instances"].to("cpu"))
            # outputs["instances"].pred_boxes.tensor[:] = 0

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

            # cv2.imwrite(prefix, v_p.get_image()[:, :, ::-1])

            masks = outputs["instances"].pred_masks
            # print(masks)
            # exit()
            masks = masks.permute(1, 2, 0).cpu().numpy()
            yichao_visual(im[:, :, ::-1] / 255.0, masks, prefix)

        if isnpz:
            masks = outputs["instances"].pred_masks
            scores = outputs["instances"].scores
            # print(scores)
            # exit()

            if masks.shape[0] != scores.shape[0]:
                raise ValueError("ggg")

            segmentation = torch.zeros_like(masks).sum(0)

            for i in range(masks.shape[0]):
                l = masks.shape[0] - 1 - i
                segmentation[masks[l]] = l + 1

                # old wrong version
                # segmentation[masks[i]] = i + 1

            if meta_dir_mode == "2":
                image_dir = osp.dirname(d["file_name"])
                mid_dir = osp.basename(image_dir)
                prefix = f"{cfg.OUTPUT_DIR}/npz/{mid_dir}"
                os.makedirs(prefix, exist_ok=True)
                # TODO
                prefix = f"{prefix}/{osp.basename(d['file_name'])[:-9]}"
            elif meta_dir_mode == "1":
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
    # cfg__.SOLVER.WARMUP_FACTOR = 1.0 / 100
    # cfg__.SOLVER.WARMUP_ITERS = 100

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

    # cfg__.OUTPUT_DIR = output_dir
    # trainer = DefaultTrainer(cfg__)

    trainer.resume_or_load(resume=True)
    trainer.train()


def validation(
        name, metadata, json_file, image_root, output_dir,
        ckpt, score_thresh_test, output_meta_dirs, meta_dir_mode,
):

    from eval_holicity_2d_metric import Holicity_2d_metric

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

    # eval 2d metric
    use_our_2d_metric = False
    debug = False
    if use_our_2d_metric:

        metadata = MetadataCatalog.get(name)
        dataset_dicts = DatasetCatalog.get(name)

        if debug:
            pass
            # right
            # predictions = inference_(predictor, dataset_dicts)
            #
            # coco_results = []
            # for m, instance in enumerate(predictions):
            #     res = instances_to_coco_json(instance, m)
            #     coco_results.extend(res)
            # # print(coco_results)
            # # exit()
            # # coco_results = list(itertools.chain(*[x for x in coco_results]))
            # outpath = f"{osp.dirname(json_file)}/{name}_results.json"
            # with PathManager.open(outpath, "w") as j_file:
            #     # logger.info(f"Caching annotations in COCO format: {cache_path}")
            #     json.dump(coco_results, j_file, cls=MyEncoder)
            #
            # Holicity_2d_metric(
            #     pre_seglist=None,
            #     pre_scorelist=None,
            #     output=osp.dirname(json_file),
            #     annFile=osp.basename(json_file),  # gt json file name
            #     resFile=f"{name}_results.json",  # results file name
            # )

        else:
            inference(
                cfg__, predictor,
                dataset_dicts=dataset_dicts, metadata=metadata, isnpz=True, isviz=False, meta_dir_mode=meta_dir_mode
            )

            results_root = cfg__.OUTPUT_DIR+"/npz"
            Holicity_2d_metric(
                pre_seglist=[osp.join(results_root, path + "_plan.png") for path in output_meta_dirs],
                pre_scorelist=[osp.join(results_root, path + "_plan.npz") for path in output_meta_dirs],
                output=osp.dirname(json_file),
                annFile=osp.basename(json_file),  # gt json file name
                resFile=f"{name}_results.json",  # results file name
            )

    else:

        evaluator = COCOEvaluator(name, cfg__, False, output_dir=output_dir)
        val_loader = build_detection_test_loader(cfg__, name)
        inference_on_dataset(predictor.model, val_loader, evaluator)


def predict(args):

    MetadataCatalog.get(args.pred_name).set(
        evaluator_type="coco", **args.metadata
    )

    cfg__ = make_cfg()
    cfg__.OUTPUT_DIR = args.pred_output_dir
    os.makedirs(cfg__.OUTPUT_DIR, exist_ok=True)

    # evaluating validation data set
    cfg__.MODEL.WEIGHTS = args.ckpt
    cfg__.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh_test  # set the testing threshold for this model
    predictor = DefaultPredictor(cfg__)

    metadata = MetadataCatalog.get(args.pred_name)
    dataset_dicts = [{"file_name": p} for p in args.img_dirs_list]
    # print(dataset_dicts)
    # exit()
    isnpz = False
    isviz = True

    inference(
        cfg__, predictor,
        dataset_dicts=dataset_dicts, metadata=metadata, isnpz=isnpz, isviz=isviz, meta_dir_mode=args.meta_dir_mode
    )


if __name__ == '__main__':

    args = docopt(__doc__)

    if args["--mode"] == "HoliCityV1":
        # print("Warning: HoliCityV1 noly can be work on dxl.cluster")
        # string = input("dxl.cluster ? (y/n)")

        from HoliCity import Config
        from HoliCity.Coco_format import build_holicityV1_coco_format as build_coco_format

    elif args["--mode"] == "ScanNet":
        # print("Warning: Scannet noly can be work on wx399")
        # string = input("wx399 ? (y/n)")

        from ScanNet import Config
        from ScanNet.Coco_format import build_scannet_coco_format as build_coco_format
    elif args["--mode"] == "Megadepth":
        from Megadepth import Config
    elif args["--mode"] == "SYNTHIA":
        from SYNTHIA import Config
    else:
        raise ValueError()

    config = Config()

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
        if args["--stage"] in ['test', 'test+valid', 'validhd']:
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

