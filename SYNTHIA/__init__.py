import os


class Config():

    def __init__(self, name="SYNTHIA"):
        self._name = name
        self.score_thresh_test = 0.5
        # self.ckpt = None
        # self.ckpt = "output/ScanNet_train_output/model_0099999.pth"  # pretrained on Scannet
        self.ckpt = "output/holiCity_output/model_0099999.pth"  # pretrained on HoliCityV0
        self.log = "trained_on_HoliCityV0"  # HoliCityV0  ScanNet

        self._relate()

    def _relate(self):
        self.metadata = {'thing_classes': ["P"]}
        self.meta_dir_mode = "2"
        self.output_dir = f"./output/{self._name}_{self.log}"

    def predict(self):
        self.pred_name = f"{self._name}_pred"
        self.pred_output_dir = f"{self.output_dir}/predict_log"

        # root = "/home/dxl/Data/megadepth/test"
        # img_dirs = sorted(os.listdir(root))
        self.img_dirs_list = load_test_images()


def load_test_images(
        TEST_LIST='/home/dxl/Data/PlaneRecover/tst_100.txt',
        dataset_dir="/home/dxl/Data/PlaneRecover"
):

    with open(TEST_LIST, 'r') as f:
        test_files_list = []
        test_files = f.readlines()
        for t in test_files:
            t_split = t[:-1].split()
            if t_split[0] == '22':  # seq 22 is not available in our preprocessed dataset, see README for more details
                continue
            test_files_list.append(dataset_dir + '/' + t_split[0] +'/'+ t_split[-1] + '.jpg')

        return test_files_list
