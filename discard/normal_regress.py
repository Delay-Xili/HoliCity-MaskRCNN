#!/usr/bin/env python3
"""Training and Evaluate the Neural Network
Usage:
    train.py [options] <yaml-config>
    train.py (-h | --help )

Arguments:
    yaml-config                      Path to the yaml hyper-parameter file

Options:
   -h --help                         Show this screen.
   -d --devices <devices>            Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>      Folder name [default: default-identifier]
"""

import torch
import torch.nn as nn
from torch.utils import data
import os.path as osp
import numpy as np
import cv2
from discard.config import C, M
from docopt import docopt
import os
from tqdm import tqdm
from discard.fpn import TinyNet

_max_plane = 20

normal_anchor = [[0.276385, -0.850640, 0.447215],
                [-0.723600, -0.525720, 0.447215],
                [-0.723600, 0.525720, 0.447215],
                [0.276385, 0.850640, 0.447215],
                [0.894425, 0.000000, 0.447215],
                [0.000000, 0.000000, 1.000000]]



class PlaneDataset(data.Dataset):
    def __init__(self, rootdir=None, split='train'):

        self.rootdir = rootdir
        with open(osp.join(rootdir, "newfilelist.txt"), "r") as f:
            samples = [line[:-1] for line in f.readlines()]

        if split == "train":
            with open(osp.join(rootdir, "20200222-middle-train.txt"), "r") as ft:
                check_sample = [line[:-1] for line in ft.readlines()]
        elif split == "valid":
            with open(osp.join(rootdir, "20200222-middle-test.txt"), "r") as ft:
                check_sample = [line[:-1] for line in ft.readlines()]
        elif split == "test":
            with open(osp.join(rootdir, "20200222-middle-test.txt"), "r") as ft:
                check_sample = [line[:-1] for line in ft.readlines()]

        filelist = []
        for file in samples:
            if file[:-7] in check_sample:
                filelist.append(file)
        # if split == "valid":
        #     self.filelist = filelist[600:]
        # elif split == "test":
        #     self.filelist = filelist[:600]
        self.filelist = filelist
        self.subset = split
        print(f"n{split}:", len(self.filelist))

        self.precompute_K_inv_dot_xy_1()

    def precompute_K_inv_dot_xy_1(self):

        x, y = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(1, -1, 512))
        xyz = np.stack([x, y, -np.ones_like(x)], axis=0)
        # print(xyz.shape)

        self.K_inv_dot_xy_1 = xyz

    def __getitem__(self, index):

        name = osp.join(self.rootdir, self.filelist[index])

        if os.path.exists(name+"_feat.npz"):
            with np.load(name+"_feat.npz") as npz:
                feats = npz["feats"]
                labels = npz["labels"]
                scores = npz["scores"]
        else:

            with np.load(name+"_mskrcnn_feat_1.npz") as npz:
                feat = npz["features"]
                boxes = npz["boxes"]  # XYXY-format
                boxes = boxes * (512 / 800)

            with np.load(name + "_plan.npz") as npz:
                # plane = data['plane']
                planes_ = npz['ws']

            gt_segmentation = cv2.imread(name + "_plan.png", cv2.IMREAD_ANYDEPTH)
            labels, score = self.batch_label_boxes(planes_, gt_segmentation, boxes)
            feats, labels, scores = self.filter(feat, labels, score)
            if np.sum(scores > 0) < _max_plane:
                _valid_num = np.sum(scores > 0)
                labels_d = np.linalg.norm(labels[:_valid_num], axis=1)
                labels[:_valid_num] /= labels_d.reshape(-1, 1) ** 2
            else:
                labels_d = np.linalg.norm(labels, axis=1)
                # normalize to n/d
                labels /= labels_d.reshape(-1, 1) ** 2

            np.savez_compressed(
                name + "_feat.npz",
                feats=feats,
                labels=labels,
                scores=scores
            )

        return torch.from_numpy(feats).float(), torch.from_numpy(labels).float(), torch.from_numpy(scores).float()

    def filter(self, feat, labels, score):
        idx = score > 0
        feat = feat[idx]
        labels = labels[idx]
        score = score[idx]
        if len(feat) <= _max_plane:
            new_feat = np.concatenate((feat, np.zeros((_max_plane-feat.shape[0], 256, 7, 7))), axis=0)
            new_label = np.concatenate((labels, np.zeros((_max_plane-labels.shape[0], 3))), axis=0)
            new_score = np.concatenate((score, np.zeros((_max_plane-score.shape[0], ))), axis=0)
            return new_feat, new_label, new_score
        idx = np.argsort(-score)
        new_feat = feat[idx[:_max_plane]]
        new_label = labels[idx[:_max_plane]]
        new_score = score[idx[:_max_plane]]

        return new_feat, new_label, new_score

    def labeling_box(self, plane, mask, box):

        x1, y1, x2, y2 = box
        assert x1 < x2 and y1 < y2
        valid_region = mask[int(y1):int(y2), int(x1):int(x2)]
        uni_idx = np.unique(valid_region)
        best_area, best_id = 0, 0
        # print(box)
        # print(uni_idx)
        for i in uni_idx:
            if i == 0:
                continue
            if np.sum(valid_region == i) > best_area:
                best_area = np.sum(valid_region == i)
                best_id = i

        if best_id == 0:
            # print(box)
            # print(uni_idx)
            # raise ValueError("bbb")
            return np.array([0.0, 0.0, 1.0]), 0

        return plane[best_id-1], best_area

    def batch_label_boxes(self, plane, mask, boxes):

        _plane = []
        _area = []
        for box in boxes:
            p, area = self.labeling_box(plane, mask, box)
            _plane.append(p)
            _area.append(area)
        return np.vstack(_plane), np.array(_area)

    def __len__(self):
        return len(self.filelist)


def cos_l1_losses(predicts, labels):

    cos_sim = nn.functional.cosine_similarity(predicts, labels)
    cos_loss = torch.mean(1 - cos_sim)

    l1_loss = torch.mean(torch.sum(torch.abs(predicts - labels), dim=1))

    return cos_loss, l1_loss


def train():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/ldcity.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    # pprint.pprint(C, indent=4)
    # resume_from = C.io.resume_from

    # FIXME: not deterministic
    # random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    num_gpus = args["--devices"].count(",") + 1
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    # 1. dataset
    batch_size = M.batch_size * num_gpus
    datadir = C.io.datadir
    kwargs = {
        "batch_size": batch_size,
        "num_workers": 12,
        "pin_memory": True,
        "drop_last": True,
    }

    train_loader = torch.utils.data.DataLoader(
        PlaneDataset(datadir, split="train"), shuffle=True, **kwargs
    )

    valid_loader = torch.utils.data.DataLoader(
        PlaneDataset(datadir, split="valid"), shuffle=False, **kwargs
    )

    model = TinyNet(3)
    model = nn.DataParallel(model)
    model = model.to(device)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=C.optim.lr,
        weight_decay=C.optim.weight_decay,
        amsgrad=C.optim.amsgrad,
    )

    for epoch in range(C.optim.max_epoch):
        loss_c, loss_l = 0, 0
        # statistic = []
        for i, (feats, labels, scores) in enumerate(tqdm(train_loader)):

            # if True:
            #     num = torch.sum(scores > 0)
            #     statistic.append(num.item())
            #     continue

            N, k, c, h, w = feats.shape
            feats = feats.view(N*k, c, h, w).cuda()
            labels = labels.view(N*k, 3).cuda()

            predicts = model(feats)

            c_loss, l_loss = cos_l1_losses(predicts, labels)
            loss_c += c_loss.item()
            loss_l += l_loss.item()
            loss = c_loss + l_loss
            loss.backward()
            optim.step()

            # if i % 100 == 0:
            #     print(f"{epoch:03}/{i:04}| cos_loss: {c_loss.item()} | L1_loss: {l_loss.item()}")
        print(f"Training | {epoch:03}| cos_loss: {loss_c / len(train_loader)} | L1_loss: {loss_l / len(train_loader)}")
        # statistic = np.array(statistic)
        # plt.hist(statistic, bins=20)
        # plt.savefig(f"statistic_train.png", dpi=100)
        # plt.close()
        # for i in range(5):
        #     print(f"{i}-th planes sample: ", np.sum(statistic==i))

        if epoch % 4 == 0:
            loss_c, loss_l = 0, 0
            # statistic = []
            for j, (feats, labels, scores) in enumerate(tqdm(valid_loader)):
                # if True:
                #     num = torch.sum(scores > 0)
                #     statistic.append(num.item())
                #     continue
                N, k, c, h, w = feats.shape
                feats = feats.view(N * k, c, h, w).cuda()
                labels = labels.view(N * k, 3).cuda()
                predicts = model(feats)
                c_loss, l_loss = cos_l1_losses(predicts, labels)
                loss_c += c_loss.item()
                loss_l += l_loss.item()

            # statistic = np.array(statistic)
            # plt.hist(statistic, bins=20)
            # plt.savefig(f"statistic_valid.png", dpi=100)
            # plt.close()
            # print("=====")
            # for i in range(5):
            #     print(f"{i}-th planes sample: ", np.sum(statistic == i))
            # exit()

            print(f"Validate | {epoch:03}| cos_loss: {loss_c / len(valid_loader)} | L1_loss: {loss_l / len(valid_loader)}")

            torch.save(model.state_dict(), osp.join(C.io.logdir, f"checkpoints/epoch_{epoch}.pt"))


if __name__ == '__main__':
    train()
