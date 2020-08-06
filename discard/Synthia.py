import os
import os.path as osp
import numpy as np

import torch
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances

import cv2

GT = "/home/dxl/Data/r2/*[0-9]/*[0-9]_0_GL.npz"
# GT = r"D:\Download\cluster.Interface\stereoCAD\sample\*[0-9]_0_GL.npz"
_name = ""
_H, _W = 512, 512
_plane_area = 32 * 32


def get_filelist(rootdir, testfile):

    with open(testfile, 'r') as f:
        test_files_list = []
        depth_file_list = []
        test_files = f.readlines()
        for t in test_files:
            t_split = t[:-1].split()

            # use these two lines if you use our preprocessed dataset
            if t_split[0] == '22': # seq 22 is not available in our preprocessed dataset, see README for more details
                continue
            test_files_list.append(rootdir + '/' + t_split[0] +'/'+ t_split[-1] + '.jpg' )
            depth_file_list.append(rootdir + '/' + t_split[0] +'/'+ t_split[-1] + '_depth.png')
    return test_files_list


def make_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.DATASETS.TRAIN = (_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0099999.pth")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 150000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (plane)

    # user parameters
    cfg.OUTPUT_DIR = "./output/holiCity_output"
    cfg.INPUT.MASK_FORMAT = "bitmask"  # polygon

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg



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
            image_name = image_name[:-9]

            image_dir = osp.dirname(d["file_name"])
            mid_dir = osp.basename(image_dir)
            prefix = f"{cfg.OUTPUT_DIR}/npz/{mid_dir}"
            os.makedirs(prefix, exist_ok=True)

            prefix = f"{prefix}/{image_name}"
            cv2.imwrite(f"{prefix}_plan.png", segmentation.cpu().numpy().astype(np.uint16))
            np.savez(f"{prefix}_plan.npz", scores=scores.cpu().numpy())


def build_evaluator(cfg):
    evaluator = COCOEvaluator(_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, _name)

    return evaluator, val_loader


def assign_global(name):
    global _name
    _name = name


def testing():
    name = "SYNTHIA_test"  # "HoliCity_train" "HoliCity_valid"
    assign_global(name)
    cache_pth = f"/home/dxl/Code/PlanarReconstruction/data/SYNTHIA_test_coco_format.json"
    register_coco_instances(name=_name,
                            metadata={'thing_classes': ["P"]},
                            json_file=cache_pth,
                            image_root="/home/dxl/Data/PlaneRecover")

    cfg__ = make_cfg()

    # begin train
    trainer = DefaultTrainer(cfg__)
    # trainer.resume_or_load(resume=True)
    # trainer.train()

    # predict and visualization
    # extract_roi_feature(cfg__)
    # build_predictor_vis(cfg__)
    # evaluator_metric(cfg__)

    # evaluating validation data set
    trainer.resume_or_load(resume=True)
    evaluator, val_loader = build_evaluator(cfg__)
    inference_on_dataset(trainer.model, val_loader, evaluator)


def coco_wrapper(image_dirs, splitxt):
    filelist = get_filelist(image_dirs, splitxt)

    coco_list = []
    for ff in filelist:
        coco_list.append({"file_name": ff})

    return coco_list


def predict():
    name = "SYNTHIA"  # "HoliCity_train" "HoliCity_valid"
    assign_global(name)

    # configuration
    # -------- change with your path ----------
    gt_image = "/home/dxl/Data/PlaneRecover"
    # results_root = "logs/tryV2/npz"
    out = "data"
    # ------------------------------------------

    cache_pth = f"{out}/HoliCityV2_valid_coco_format.json"
    register_coco_instances(name=_name,
                            metadata={'thing_classes': ["P"]},
                            json_file=cache_pth,
                            image_root=gt_image)

    cfg__ = make_cfg()
    cfg__.OUTPUT_DIR = f"./output/{_name}_output"

    # begin train
    # trainer = DefaultTrainer(cfg__)
    # trainer.resume_or_load(resume=True)
    # trainer.train()

    # evaluating validation data set
    cfg__.MODEL.WEIGHTS = os.path.join("./output/holiCity_output", "model_0099999.pth")
    cfg__.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg__.DATASETS.TEST = (_name,)
    metadata = MetadataCatalog.get(_name)
    dataset_dicts = coco_wrapper(gt_image, "/home/dxl/Data/PlaneRecover/tst_100.txt")

    save_npz(cfg__, dataset_dicts=dataset_dicts, metadata=metadata, isnpz=False, isviz=True)


if __name__ == '__main__':

    predict()
