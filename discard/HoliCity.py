import os
import os.path as osp
import numpy as np

import torch
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
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


def get_filelist(rootdir, split):
    with open(osp.join(rootdir, "newfilelist.txt"), "r") as f:
        samples = [line[:-1] for line in f.readlines()]

    if split == "train":
        with open(osp.join(rootdir, "20200222-middle-train.txt"), "r") as ft:
            check_sample = [line[:-1] for line in ft.readlines()]
    elif split == "valid":
        with open(osp.join(rootdir, "20200222-middle-test.txt"), "r") as ft:
            check_sample = [line[:-1] for line in ft.readlines()]

    filelist = []
    for file in samples:
        if file[:-7] in check_sample:
            filelist.append(osp.join(rootdir, file))

    print(f"n{split}:", len(filelist))

    return filelist


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


def build_predictor_vis(cfg):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0099999.pth")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg.DATASETS.TEST = (_name,)
    predictor = DefaultPredictor(cfg)
    # print(predictor.model.roi)
    # exit()

    dataset_dicts = DatasetCatalog.get(_name)
    metadata = MetadataCatalog.get(_name)
    # print(dataset_dicts)
    for k in range(500):
        d = dataset_dicts[k]
        im = cv2.imread(d["file_name"])

        outputs = predictor(im)
        # exit()
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
        v_p_nobbx = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=2,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )

        v_p = v_p.draw_instance_predictions(outputs["instances"].to("cpu"))
        v_gt = v_gt.draw_dataset_dict(d)
        outputs["instances"].pred_boxes.tensor[:] = 0
        v_p_nobbx = v_p_nobbx.draw_instance_predictions(outputs["instances"].to("cpu"))

        # plt.figure(figsize=[30, 10])
        # plt.subplot(131), plt.imshow(v_p_nobbx.get_image()[:, :, ::-1])
        # plt.subplot(132), plt.imshow(v_p.get_image()[:, :, ::-1])
        # plt.subplot(133), plt.imshow(v_gt.get_image()[:, :, ::-1])
        cv2.imwrite(f"demo/predict_{d['image_id']}_im.png", im)
        # cv2.imwrite(f"demo/predict_{d['image_id']}_ps.png", v_p_nobbx.get_image())
        cv2.imwrite(f"demo/predict_{d['image_id']}_pd.png", v_p.get_image())
        cv2.imwrite(f"demo/predict_{d['image_id']}_gt.png", v_gt.get_image())
        # plt.imshow(v_p_nobbx.get_image()[:, :, ::-1])
        # plt.savefig(f"demo/predict_{d['image_id']}_ps.png", dpi=100), plt.close()
        # plt.imshow(v_p.get_image()[:, :, ::-1])
        # plt.savefig(f"demo/predict_{d['image_id']}_pb.png", dpi=100), plt.close()
        # plt.imshow(v_gt.get_image()[:, :, ::-1])
        # plt.savefig(f"demo/predict_{d['image_id']}_gt.png", dpi=100), plt.close()
        print(f"Saving demo/predict_{d['image_id']}.png")


def build_evaluator(cfg):
    evaluator = COCOEvaluator(_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, _name)

    return evaluator, val_loader


def assign_global(name):
    global _name
    _name = name


# def SYNTHIA():
#     name = "SYNTHIA_test"  # "HoliCity_train" "HoliCity_valid"
#     assign_global(name)
#     cache_pth = f"/home/dxl/Code/PlanarReconstruction/data/SYNTHIA_test_coco_format.json"
#     register_coco_instances(name=_name,
#                             metadata={'thing_classes': ["P"]},
#                             json_file=cache_pth,
#                             image_root="/home/dxl/Data/PlaneRecover")
#
#     cfg__ = make_cfg()
#
#     # begin train
#     trainer = DefaultTrainer(cfg__)
#     # trainer.resume_or_load(resume=True)
#     # trainer.train()
#
#     # predict and visualization
#     # extract_roi_feature(cfg__)
#     # build_predictor_vis(cfg__)
#     # evaluator_metric(cfg__)
#
#     # evaluating validation data set
#     trainer.resume_or_load(resume=True)
#     evaluator, val_loader = build_evaluator(cfg__)
#     inference_on_dataset(trainer.model, val_loader, evaluator)


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


def HolicityV2():
    name = "HoliCity_valid_scannet_pretrained"  # "HoliCity_train" "HoliCity_valid"
    assign_global(name)

    # configuration
    # -------- change with your path ----------
    gt_image = "/home/dxl/Data/LondonCity/V2/train-images"  # train-images, valid-test-images
    # gt_label = "/home/dxl/Data/LondonCity/V2/valid-test-GT"
    # name = "HoliCityV2"
    out = "data"
    # ------------------------------------------

    # filelist = V2_filelist(gt_image)
    # img_dirs = [osp.join(gt_image, path) for path in filelist]
    # gt_seglist = [osp.join(gt_label, path.replace("_imag.jpg", "_plan.png")) for path in filelist]

    cache_pth = f"{out}/HoliCityV2_valid_coco_format.json"
    # if not os.path.isfile(cache_pth):
    #     cache_pth = COCO_format(img_dirs, gt_seglist, out, f"{_name}")
        # exit()

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

    # predict and visualization
    # extract_roi_feature(cfg__)
    # build_predictor_vis(cfg__)
    # evaluator_metric(cfg__)

    # evaluating validation data set
    # trainer.resume_or_load(resume=True)
    cfg__.MODEL.WEIGHTS = os.path.join("./output/ScanNet_train_output", "model_0099999.pth")
    cfg__.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg__.DATASETS.TEST = (_name,)
    metadata = MetadataCatalog.get(_name)
    dataset_dicts = coco_wrapper(gt_image, "/home/dxl/Data/LondonCity/V2/test-hd-sub.txt")
    # dataset_dicts = DatasetCatalog.get(_name)

    save_npz(cfg__, dataset_dicts=dataset_dicts, metadata=metadata, isnpz=False, isviz=True)
    exit()

    predictor = DefaultPredictor(cfg__)
    evaluator, val_loader = build_evaluator(cfg__)
    inference_on_dataset(predictor.model, val_loader, evaluator)


def coco_wrapper(image_dirs, splitxt):
    filelist = get_image_list(image_dirs, splitxt)

    coco_list = []
    for ff in filelist:
        coco_list.append({"file_name": ff})

    return coco_list


def get_image_list(image_dirs, splitxt):

    with open(splitxt, "r") as f:
        samples = [line[:-1] for line in f.readlines()]

    middle_p = sorted(os.listdir(image_dirs))
    filelist = []
    for pth in middle_p:
        imgs = sorted(os.listdir(osp.join(image_dirs, pth)))
        for img in imgs:
            img_pth = osp.join(pth, img)
            if img_pth[:-16] in samples:
                filelist.append(osp.join(image_dirs, pth, img))  # abs path

    return filelist


if __name__ == '__main__':
    HolicityV2()
    # SYNTHIA()
