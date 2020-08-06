import os
import numpy as np
import json
import datetime
from fvcore.common.file_io import PathManager

from tqdm import tqdm
import cv2
from pycocotools import mask


_H, _W = 512, 512


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


def get_one_coco_image(id, im_path):
    coco_image = {
        "id": id,
        "width": _W,
        "height": _H,
        "file_name": im_path,
    }
    return coco_image


def get_one_coco_annotation(gt_seg_npz, num_planes, coco_image_id, annotations_len):
    unval = np.unique(gt_seg_npz)
    # print(f"{root}/{gt_seg_path}")
    # print(unval)
    # exit()

    # area = []
    # for val in unval[1:]:
    #     gt_mask = np.asarray(gt_seg_npz == val, order="F")
    #     encoded_gt = mask.encode(gt_mask)
    #     area_gt = mask.area(encoded_gt)
    #     area.append(area_gt)

    # area = sorted(area)
    # top5 = area[-5] if len(area) > 6 else 0
    # print(sorted(area))
    # exit()

    k = 1
    coco_annotations = []
    # print(num_planes)
    for val in range(num_planes):
        coco_annotation = {}
        gt_mask = gt_seg_npz == val
        gt_mask = cv2.erode(gt_mask.astype(np.uint8), np.ones((3, 3)))
        gt_mask = np.asarray(gt_mask, order="F")  # without order="F", mask.encode will drop an error.
        encoded_gt = mask.encode(gt_mask)

        area_gt = mask.area(encoded_gt)
        bbox_gt = toBbox(gt_mask)

        # if area_gt < _plane_area:
        #     continue

        # if area_gt < top5:
        #     continue

        coco_annotation["id"] = annotations_len + k
        coco_annotation["image_id"] = coco_image_id
        coco_annotation["bbox"] = [round(float(x), 3) for x in bbox_gt]
        coco_annotation["segmentation"] = encoded_gt
        coco_annotation["area"] = area_gt
        coco_annotation["category_id"] = 0
        coco_annotation["iscrowd"] = 0
        k += 1
        coco_annotations.append(coco_annotation)

    return coco_annotations


def save_coco_json(categories, coco_images, coco_annotations, output_folder, dataset_name):

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

    os.makedirs(output_folder, exist_ok=True)

    cache_path = os.path.join(output_folder, f"{dataset_name}_coco_format.json")

    with PathManager.open(cache_path, "w") as json_file:
        # logger.info(f"Caching annotations in COCO format: {cache_path}")
        json.dump(coco_dict, json_file, cls=MyEncoder)

    return cache_path


def convert_planenet2coco(root_npz, splitxt, output_img_folder, output_folder, dataset_name):

    categories = [
        {"id": 0, "name": "P"}
    ]

    with open(splitxt, "r") as f:
        samples = [line[:-1] for line in f.readlines()]

    os.makedirs(output_img_folder, exist_ok=True)

    coco_images = []
    coco_annotations = []
    for i, npz_name in tqdm(enumerate(samples)):

        with np.load(f"{root_npz}/{npz_name}") as npz:
            image = npz['image']
            gt_seg = npz['segmentation']
            num_p = npz['num_planes'][0]

        im_path = f"{output_img_folder}/{npz_name.replace('.npz', '.jpg')}"
        image = cv2.resize(image, (_H, _W))
        cv2.imwrite(im_path, image)
        coco_images.append(get_one_coco_image(i, im_path))

        gt_seg = cv2.resize(gt_seg, (_H, _W))
        coco_annotations += get_one_coco_annotation(gt_seg, num_p, i, len(coco_annotations))

    save_coco_json(categories, coco_images, coco_annotations, output_folder, dataset_name)


def build_scannet_coco_format(name="ScanNet_train"):

    # name = "ScanNet_train"  # "ScanNet_train" "ScanNet_val"
    json_out = "data/ScanNet"
    # ------------------------------------------

    split = name.split('_')[1]
    root_npz = f"dataset/scannet/{split}"
    splitxt = f"dataset/scannet/{split}.txt"
    output_img_folder = f"dataset/scannet/{split}_image"

    convert_planenet2coco(
        root_npz, splitxt, output_img_folder, output_folder=json_out, dataset_name=name,
    )


def build_scannet_train():
    build_scannet_coco_format(name="ScanNet_train")


def build_scannet_val():
    build_scannet_coco_format(name="ScanNet_val")


# if __name__ == '__main__':
#
#     COCO_format()
