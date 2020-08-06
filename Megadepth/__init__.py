import os


class Config():

    def __init__(self, name="Megadepth"):
        self._name = name
        self.score_thresh_test = 0.5
        # self.ckpt = None
        # self.ckpt = "output/ScanNet_train_output/model_0099999.pth"  # pretrained on Scannet
        self.ckpt = "output/holiCity_output/model_0099999.pth"  # pretrained on HoliCityV0
        self.log = "trained_on_HoliCityV0"  # HoliCityV0  ScanNet

        self._relate()

    def _relate(self):
        self.metadata = {'thing_classes': ["P"]}
        self.meta_dir_mode = "1"
        self.output_dir = f"./output/{self._name}_{self.log}"

    # def training(self):
    #     self.train_name = f"{self._name}_train"
    #     self.train_output_dir = f"{self.output_dir}/train_log"
    #     self.train_json_file = f"{self.train_name}_coco_format.json"

    def predict(self):
        self.pred_name = f"{self._name}_pred"
        self.pred_output_dir = f"{self.output_dir}/predict_log"

        root = "/home/dxl/Data/megadepth/test"
        img_dirs = sorted(os.listdir(root))
        self.img_dirs_list = [os.path.join(root, p) for p in img_dirs]

