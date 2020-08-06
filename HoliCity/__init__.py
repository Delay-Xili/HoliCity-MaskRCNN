from .utils import V1_filelist


class Config():

    def __init__(self, name="HoliCityV1"):
        self.machine = "w399"   # dxl, w399
        self._name = name
        self.split_version = "v1"
        if self.machine == "dxl.cluster":
            self.split_version = 'v1'

        self.score_thresh_test = 0.5
        self.ckpt = None
        # self.ckpt = "output/ScanNet_train_output/model_0099999.pth"  # pretrained on Scannet
        # self.ckpt = "output/HoliCityV0/model_0099999.pth"  # pretrained on HoliCityV0
        self.log = "lr2.5e4-lrdecay10w"  # HoliCityV0  ScanNet
        self.lr = 2.5e-4
        self.lr_decay = 100000

        self._relate()

    def _relate(self):
        self.metadata = {'thing_classes': ["P"]}
        self.meta_dir_mode = "2"
        self.output_dir = f"./output/{self._name}_{self.split_version}_{self.log}"
        self.json_out = f"data/HoliCityV1_{self.split_version}"
        self.eval_period = 5000
        if self.machine == "w399":
            self.image_root = "./"
            self.root = "dataset/HoliCityV1"
        elif self.machine == "dxl.cluster":
            self.image_root = "/home/dxl/Data/LondonCity/V1/image"
            self.root = "/home/dxl/Data/LondonCity/V1"
        else:
            raise ValueError()

    def training(self):
        self.train_name = f"{self._name}_train"
        self.train_output_dir = f"{self.output_dir}/train_log"
        self.train_json_file = f"{self.train_name}_coco_format.json"

    def valid(self, postfix):
        if postfix not in ['valid', 'test', 'test+valid', 'validhd', 'validld']:
            raise ValueError()
        self.val_name = f"{self._name}_{postfix}"
        self.val_output_dir = f"{self.output_dir}/{postfix}_log"
        self.val_json_file = f"{self.val_name}_coco_format.json"
        self.valid_meta_dirs = V1_filelist(split=postfix, rootdir=self.root, split_version=self.split_version)

    def predict(self):
        self.pred_name = f"{self._name}_pred"
        self.pred_output_dir = f"{self.output_dir}/predict_log"

        self.img_dirs_list = get_predict_img(self.image_root)


def get_predict_img(image_root):
    # some high-resolution samples for visualization
    with open("/home/dxl/Data/LondonCity/V1/split/v1/filelist.txt", "r") as f:
        filelist = [line[:-1] for line in f.readlines()]

    with open("/home/dxl/Data/LondonCity/V1/split/v1/best-hd.txt", "r") as f:
        samples = [line[:-1] for line in f.readlines()]
    # print(samples)
    flist = []
    for pth in filelist:
        if pth[:-7] in samples:
            flist.append(pth)

    return [f"{image_root}/{p}_imag.jpg" for p in flist]

