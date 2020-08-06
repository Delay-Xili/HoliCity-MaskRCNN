import os


class Config():

    def __init__(self, name="ScanNet"):
        self._name = name
        self.json_out = "data/ScanNet"
        self.image_root = "./"
        self.metadata = {'thing_classes': ["P"]}
        self.eval_period = 5000
        self.score_thresh_test = 0.5
        self.ckpt = "output/ScanNet_train_output/model_0099999.pth"
        self.meta_dir_mode = "1"

    def training(self):
        self.train_name = f"{self._name}_train"
        self.train_output_dir = f"./output/{self.train_name}_output"
        self.train_json_file = f"{self.train_name}_coco_format.json"

    def valid(self, postfix):
        self.val_name = f"{self._name}_{postfix}"
        self.val_output_dir = f"./output/{self.val_name}_output"
        self.val_json_file = f"{self.val_name}_coco_format.json"

        with open(f"dataset/scannet/{postfix}.txt", "r") as f:
            meta_dir = [line[:-1] for line in f.readlines()]
        self.valid_meta_dirs = [p[:-4] for p in meta_dir]

    def predict(self):
        self.pred_name = f"{self._name}_pred"
        self.pred_output_dir = f"./output/{self.pred_name}_output"

        self.img_dirs_list = []


