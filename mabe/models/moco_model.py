import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm

from mabe.archs import define_network
from mabe.data.transform import TransformsSimCLR
from mabe.losses import cross_entropy_loss, info_nce_loss
from mabe.models.base_model import BaseModel
from mabe.simclr.modules import LARS
from mabe.utils import get_root_logger, master_only


class MOCOModel(BaseModel):
    def __init__(self, opt):
        super(MOCOModel, self).__init__(opt)

        # define network
        self.net = define_network(deepcopy(opt["network"]))
        self.net = self.model_to_device(self.net)
        self.print_network(self.net)

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network", None)
        if load_path is not None:
            self.load_network(
                self.net, load_path, self.opt["path"].get("strict_load", True)
            )

        self.init_training_settings()

        self.scaler = GradScaler()

        frame_size = opt["common"]["frame_size"]
        pretrained = True
        num_prev_frames = opt["common"]["num_prev_frames"]
        num_next_frames = opt["common"]["num_next_frames"]
        n_channel = num_prev_frames + num_next_frames + 1
        self.transform_train = TransformsSimCLR(
            frame_size, pretrained, n_channel, train=True
        )
        self.transform_train_td = TransformsSimCLR(
            frame_size, pretrained, n_channel, train="td"
        )
        self.transform_val = TransformsSimCLR(
            frame_size, pretrained, n_channel, train=False
        )

    def init_training_settings(self):
        self.net.train()
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_type = train_opt["optim"].pop("type")
        if optim_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), **train_opt["optim"]
            )
        elif optim_type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(), **train_opt["optim"]
            )
        elif optim_type == "LARS":
            self.optimizer = LARS(self.net.parameters(), **train_opt["optim"])
        elif optim_type == "SGD":
            self.optimizer = torch.optim.SGD(
                self.net.parameters(), **train_opt["optim"]
            )
        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supperted yet.")
        self.optimizers.append(self.optimizer)

    def feed_data(self, data, train):
        self.idx = data["idx"].to(self.device, non_blocking=True)
        self.seq_id = data["seq_id"].to(self.device, non_blocking=True)
        self.seq_id = data["seq_id"].to(self.device, non_blocking=True)
        x1 = data["x1"].to(self.device, non_blocking=True)
        x2 = data["x2"].to(self.device, non_blocking=True)
        x1 = x1.float() / 255.0
        x2 = x2.float() / 255.0
        if train:
            self.x11 = self.transform_train(x1)
            self.x12 = self.transform_train_td(x1)
            self.x21 = self.transform_train(x2)
            self.x22 = self.transform_train_td(x2)
        else:
            self.x11 = self.transform_val(x1)
            self.x12 = self.x11
            self.x21 = self.x11
            self.x22 = self.x11
        if "label" in data:
            self.label = data["label"].to(self.device, non_blocking=True)

    def optimize_parameters_amp(self, current_iter):
        self.optimizer.zero_grad()

        with autocast():
            l_total = 0
            loss_dict = OrderedDict()

            logits, labels = self.net(
                self.x11, self.x12, self.x21, self.x22, self.seq_id
            )
            l_intra, l_inter = cross_entropy_loss(
                logits, labels, inter_split=logits.shape[0] // 3
            )
            l_total += l_intra
            loss_dict["l_intra"] = l_intra
            l_inter = l_inter * 1
            l_total += l_inter
            loss_dict["l_inter"] = l_inter
            # loss_dict["temperature"] = self.net.module.T

        self.scaler.scale(l_total).backward()
        # self.scaler.unscale_(self.optimizer)
        # torch.nn.utils.clip_grad_norm_(
        #     self.net.parameters(), self.opt["train"]["grad_norm_clip"]
        # )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def optimize_parameters(self, current_iter):
        self.optimizer.zero_grad()

        l_total = 0
        loss_dict = OrderedDict()

        logits, labels = self.net(self.x11, self.x12, self.x21, self.x22)
        l_intra, l_inter = cross_entropy_loss(
            logits, labels, inter_split=logits.shape[0] // 3, inter_weight=0.1
        )
        l_total += l_intra
        loss_dict["l_intra"] = l_intra
        l_inter = l_inter * 0.1
        # l_inter = l_inter * self.adjust_weight(current_iter)
        l_total += l_inter
        loss_dict["l_inter"] = l_inter
        loss_dict["temperature"] = self.net.module.T

        l_total.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     self.net.parameters(), self.opt["train"]["grad_norm_clip"]
        # )
        self.optimizer.step()

        self.net.module.T.data = torch.clamp(self.net.module.T.data, 0, 4.6052)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def adjust_weight(self, current_iter):
        if current_iter <= 50000:
            return 0.1
        elif current_iter <= 70000:
            return 0.1 * (current_iter - 50000) / 20000
        return 0

    @torch.no_grad()
    def test(self, dataset, dataloader):
        self.net.eval()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        idxs = []
        feats = []
        labels = []

        for data in tqdm(dataloader):
            self.feed_data(data, train=False)
            idxs.append(self.idx)

            output = self.net(self.x11, self.x12, self.x21, self.x22, self.seq_id)
            feat = output
            feats.append(feat)

            has_label = "label" in data
            if has_label:
                labels.append(self.label)

        idxs = torch.cat(idxs)
        print(1, rank, idxs.shape, idxs[:10])
        feats = torch.cat(feats)
        print(1, rank, feats.shape)
        if has_label:
            labels = torch.cat(labels)
            print(1, rank, labels.shape)
        dist.barrier()

        idxs_list = [torch.zeros_like(idxs) for _ in range(world_size)]
        dist.all_gather(idxs_list, idxs)
        idxs = torch.stack(idxs_list, dim=1).flatten(0, 1)
        print(2, rank, idxs.shape, idxs[:10])
        feats_list = [torch.zeros_like(feats) for _ in range(world_size)]
        dist.all_gather(feats_list, feats)
        feats = torch.stack(feats_list, dim=1).flatten(0, 1)
        print(2, rank, feats.shape)
        if has_label:
            labels_list = [torch.zeros_like(labels) for _ in range(world_size)]
            dist.all_gather(labels_list, labels)
            labels = torch.stack(labels_list, dim=1).flatten(0, 1)
            print(2, rank, labels.shape)
        if rank == 0:
            self.feats = feats
            self.feats_npy = self.feats.cpu().numpy()
            assert validate_submission(self.feats_npy, dataset.frame_number_map)
            self.labels = None
            self.labels_npy = None
            if has_label:
                self.labels = labels
                self.labels_npy = self.labels.cpu().numpy()

        self.net.train()

    def save(self, epoch, current_iter):
        self.save_network(self.net, "net", current_iter)
        # self.save_training_state(epoch, current_iter)

    @master_only
    def save_result(self, epoch, current_iter, label):
        if current_iter == -1:
            current_iter = "latest"
        model_filename = f"net_{current_iter}.pth"
        model_path = os.path.join(self.opt["path"]["models"], model_filename)

        feats_path = model_path.replace(".pth", f"_{label}_feats.npy")
        np.save(feats_path, self.feats_npy)

        if self.labels is not None:
            labels_path = model_path.replace(".pth", f"_{label}_labels.npy")
            np.save(labels_path, self.labels_npy)


def validate_submission(submission, frame_number_map):
    if not isinstance(submission, np.ndarray):
        print("Embeddings should be a numpy array")
        return False
    elif not len(submission.shape) == 2:
        print("Embeddings should be 2D array")
        return False
    elif not submission.shape[1] <= 128:
        print("Embeddings too large, max allowed is 128")
        return False
    elif not isinstance(submission[0, 0], np.float32):
        print(f"Embeddings are not float32")
        return False

    total_clip_length = frame_number_map[list(frame_number_map.keys())[-1]][1]

    if not len(submission) == total_clip_length:
        print(f"Emebddings length doesn't match submission clips total length")
        return False

    if not np.isfinite(submission).all():
        print(f"Emebddings contains NaN or infinity")
        return False

    print("All checks passed")
    return True
