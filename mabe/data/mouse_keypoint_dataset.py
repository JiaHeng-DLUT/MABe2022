import numpy as np
import torch
from torch.utils.data import Dataset

from mabe.utils import get_root_logger


class MouseKeypointDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        data_path_list = opt["data_path_list"]
        meta_path_list = opt["meta_path_list"]
        # Number of frames per clip
        self.num_frames = opt["num_frames"]
        # Number of clips per video
        self.num_clips = int(opt["total_frames"] / opt["num_frames"])
        self.mean = torch.Tensor(opt["mean"])
        self.std = torch.Tensor(opt["std"])
        self.use_label = opt["use_label"]

        data = {}
        for data_path in data_path_list:
            data = {**data, **np.load(data_path, allow_pickle=True).item()["sequences"]}
        self.seqs = data

        data = []
        for meta_path in meta_path_list:
            data = data + open(meta_path).readlines()
        data = [x.strip() for x in data]
        self.id_list = data

        self.frame_number_map = np.load(
            opt["frame_number_map_path"], allow_pickle=True
        ).item()
        self.video_names = list(self.frame_number_map.keys())
        # IMPORTANT: the frame number map should be sorted
        frame_nums = np.array([self.frame_number_map[k] for k in self.video_names])
        assert np.all(np.diff(frame_nums[:, 0]) > 0), "Frame number map is not sorted"

        logger = get_root_logger()
        self.hflip_prob = opt["hflip_prob"]
        logger.info(f"hflip_prob: {self.hflip_prob}")
        self.vflip_prob = opt["vflip_prob"]
        logger.info(f"vflip_prob: {self.vflip_prob}")
        self.htrans_prob = opt["htrans_prob"]
        logger.info(f"htrans_prob: {self.htrans_prob}")
        self.vtrans_prob = opt["vtrans_prob"]
        logger.info(f"vtrans_prob: {self.vtrans_prob}")
        self.max_trans = opt["max_trans"]
        logger.info(f"max_trans: {self.max_trans}")
        self.rot_prob = opt["rot_prob"]
        logger.info(f"rot_prob: {self.rot_prob}")

    def __getitem__(self, index):
        ret = {}

        ID = self.id_list[index // self.num_clips]
        ret.update({"ID": ID})

        pos = index % self.num_clips * self.num_frames
        ret.update({"pos": pos})

        # (num_frames, num_animals, num_keypoints, 2)
        keypoints = torch.from_numpy(
            self.seqs[ID]["keypoints"][pos : pos + self.num_frames]
        )

        # data augmentations
        if self.hflip_prob is not None:
            if np.random.uniform() < self.hflip_prob:
                # (-x, y), (-cos, sin)
                keypoints[:, :, :, 0] = -keypoints[:, :, :, 0]
        if self.vflip_prob is not None:
            if np.random.uniform() < self.vflip_prob:
                # (x, -y), (cos, -sin)
                keypoints[:, :, :, 1] = -keypoints[:, :, :, 1]
        if self.htrans_prob is not None:
            if np.random.uniform() < self.htrans_prob:
                h_translation = np.random.uniform(
                    low=-self.max_trans, high=self.max_trans
                )
                keypoints[:, :, :, 0] += h_translation
        if self.vtrans_prob is not None:
            if np.random.uniform() < self.vtrans_prob:
                v_translation = np.random.uniform(
                    low=-self.max_trans, high=self.max_trans
                )
                keypoints[:, :, :, 1] += v_translation
        if self.rot_prob is not None:
            if np.random.uniform() < self.rot_prob:
                rotation = np.random.uniform(low=-np.pi, high=np.pi)
                R = torch.Tensor(
                    [
                        [np.cos(rotation), -np.sin(rotation)],
                        [np.sin(rotation), np.cos(rotation)],
                    ]
                )
                keypoints = keypoints @ R

        # (num_frames, num_animals, num_keypoints * 2)
        keypoints = torch.flatten(keypoints, 2)
        # (num_frames, num_animals)
        mask = ~(torch.isnan(keypoints).long().sum(-1).bool())
        ret.update({"mask": mask})

        keypoints = (keypoints - self.mean) / self.std
        keypoints = torch.nan_to_num(keypoints, nan=0)
        ret.update({"keypoints": keypoints})

        if self.use_label:
            labels = torch.from_numpy(
                self.seqs[ID]["annotations"][:, pos : pos + self.num_frames]
            ).T
            labels = torch.nan_to_num(labels, nan=-100)
            ret.update({"labels": labels})

        return ret

    def __len__(self):
        return len(self.id_list) * self.num_clips
