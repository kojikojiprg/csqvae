import os
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

training_ids = [
    1,
    2,
    4,
    5,
    8,
    9,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    25,
    27,
    28,
    31,
    34,
    35,
    38,
    45,
    46,
    47,
    49,
    50,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    70,
    74,
    78,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    89,
    91,
    92,
    93,
    94,
    95,
    97,
    98,
    100,
    103,
]


class NTU_RGBD(torch.utils.data.Dataset):
    def __init__(self, root, config, is_train, split_type="cross_subject"):
        super().__init__()
        self.root = root
        self.ma_w = config.ma_w
        self.is_train = is_train
        self.split_type = split_type

        self.x = []
        self.seq_len_lst = []
        self.labels = []

        self.create()

    def sid_is_train(self, sid):
        if self.split_type == "cross_subject":
            return sid in training_ids
        elif self.split_type == "cross_suetup":
            return sid % 2 == 0

    def load(self):
        skeletons_dir = os.path.join(self.root, "nturgb+d_skeletons")
        files = sorted(glob(os.path.join(skeletons_dir, "*.skeleton")))

        data_dicts = []
        for path in tqdm(files, ncols=100, desc="loading"):
            file_name = os.path.basename(path).replace(".skeleton", "")

            subject_id = int(file_name[9:12])
            if self.is_train and not self.sid_is_train(subject_id):
                continue
            elif not self.is_train and self.sid_is_train(subject_id):
                continue

            label = int(file_name[-3:])
            if label not in list(range(1, 31)):
                continue

            data_dict = _read_skeleton(path)
            data_dicts.append(data_dict)

        return data_dicts

    @staticmethod
    def moving_agerage(x, w):
        seq_len, pt, d = x.shape
        x = x.reshape(seq_len, -1)

        new_x = []
        for i in range(pt * d):
            new_x.append(np.convolve(x[:, i], np.ones(w), "valid").reshape(-1, 1) / w)
        x = np.concatenate(new_x, axis=1).reshape(-1, pt, d)
        return x[w:-w]  # remove prepend and append

    # @staticmethod
    # def cleansing(x, v_min):
    #     seq_len, pt, d = x.shape
    #     v = np.diff(x, axis=0)
    #     mask = np.all(v < v_min, axis=2)

    @staticmethod
    def pad_skeleton_seq(skel_seq, length=300):
        return np.pad(
            skel_seq,
            ((0, length - len(skel_seq)), (0, 0), (0, 0)),
            "constant",
            constant_values=np.nan,
        )

    @staticmethod
    def min_max_sacaling(x):
        d = x.shape[-1]
        x_min = np.nanmin(x, axis=(0, 1, 2)).reshape(1, 1, 1, d)
        x_max = np.nanmax(x, axis=(0, 1, 2)).reshape(1, 1, 1, d)
        return (x - x_min) / (x_max - x_min), x_min, x_max

    def create(self):
        data_dicts = self.load()

        for data_dict in tqdm(data_dicts, ncols=100, desc="creating"):
            label = data_dict["label"]
            for key, val in data_dict.items():
                if "skel" in key:
                    self.labels.append(label)
                    val = self.moving_agerage(val, self.ma_w)
                    self.seq_len_lst.append(len(val))
                    val = self.pad_skeleton_seq(val)
                    self.x.append(val.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.x[idx]
        seq_len = self.seq_len_lst[idx]
        label = self.labels[idx]
        return x, seq_len, label


def _read_skeleton(file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True):
    f = open(file_path, "r")
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    file_name = os.path.basename(file_path).replace(".skeleton", "")
    label = int(file_name[-3:])
    nframe = int(datas[0][:-1])
    bodymat = {"label": label, "nframe": nframe, "nbodys": [], "njoints": njoints}
    for body in range(max_body):
        if save_skelxyz:
            bodymat["skel_body{}".format(body)] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat["rgb_body{}".format(body)] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat["depth_body{}".format(body)] = np.zeros(shape=(nframe, njoints, 2))
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])
        if bodycount == 0:
            continue
        # skip the empty frame
        bodymat["nbodys"].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = "skel_body{}".format(body)
            rgb_body = "rgb_body{}".format(body)
            depth_body = "depth_body{}".format(body)

            cursor += 1

            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(" ")
                jointinfo = np.array(list(map(float, jointinfo)))
                if save_skelxyz:
                    bodymat[skel_body][frame, joint] = jointinfo[:3]
                if save_depthxy:
                    bodymat[depth_body][frame, joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][frame, joint] = jointinfo[5:7]

    # prune the abundant bodys
    for each in range(max_body):
        if each >= max(bodymat["nbodys"]):
            if save_skelxyz:
                del bodymat["skel_body{}".format(each)]
            if save_rgbxy:
                del bodymat["rgb_body{}".format(each)]
            if save_depthxy:
                del bodymat["depth_body{}".format(each)]
    return bodymat
