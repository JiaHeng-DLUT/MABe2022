import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from simclr.modules import LARS
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm

from mabe.archs import define_network
from mabe.models.base_model import BaseModel
from mabe.utils import get_root_logger, master_only


class GPTModel(BaseModel):
    def __init__(self, opt):
        super(GPTModel, self).__init__(opt)

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

    def init_training_settings(self):
        self.net.train()
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        """
        From: https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136

        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            torch.nn.BatchNorm2d,
        )
        for mn, m in self.net.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.net.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params)
        )

        train_opt = self.opt["train"]

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_opt["optim"]["weight_decay"],
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optim_type = train_opt["optim"].pop("type")
        if optim_type == "Adam":
            self.optimizer = torch.optim.Adam(optim_groups, **train_opt["optim"])
        elif optim_type == "AdamW":
            self.optimizer = torch.optim.AdamW(optim_groups, **train_opt["optim"])
        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supperted yet.")
        self.optimizers.append(self.optimizer)

    def feed_data(self, data, train):
        self.tokens = data["keypoints"].to(self.device)
        self.mask = data["mask"].to(self.device).long()
        self.pos = data["pos"]
        if "labels" in data:
            self.labels = data["labels"].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer.zero_grad()
        decode_object = self.opt["train"]["decode_object"]
        decode_frame = self.opt["train"]["decode_frame"]
        flip = self.opt["train"]["flip"]
        with autocast():
            self.output = self.net(
                self.tokens,
                self.mask,
                self.pos,
                flip=flip,
                decode_object=decode_object,
                decode_frame=decode_frame,
            )

            l_total = 0
            loss_dict = OrderedDict()

            if decode_object:
                l_animal_LR = F.mse_loss(
                    self.output["animal_LR"], torch.zeros_like(self.output["animal_LR"])
                )
                l_total += l_animal_LR
                loss_dict["l_animal_LR"] = l_animal_LR
                if flip:
                    l_animal_RL = F.mse_loss(
                        self.output["animal_RL"],
                        torch.zeros_like(self.output["animal_RL"]),
                    )
                    l_total += l_animal_RL
                    loss_dict["l_animal_RL"] = l_animal_RL

            if decode_frame:
                l_frame_LR = F.mse_loss(
                    self.output["frame_LR"], torch.zeros_like(self.output["frame_LR"])
                )
                l_total += l_frame_LR
                loss_dict["l_frame_LR"] = l_frame_LR
                if flip:
                    l_frame_RL = F.mse_loss(
                        self.output["frame_RL"],
                        torch.zeros_like(self.output["frame_RL"]),
                    )
                    l_total += l_frame_RL
                    loss_dict["l_frame_RL"] = l_frame_RL

        self.scaler.scale(l_total).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), self.opt["train"]["grad_norm_clip"]
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    @torch.no_grad()
    def test(self, dataset, dataloader):
        self.net.eval()
        N = len(dataset)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert N % world_size == 0
        feats = []
        labels = []

        for data in tqdm(dataloader):
            self.feed_data(data, train=False)

            output = self.net(self.tokens, self.mask, self.pos)

            feat = output["feat_LR"]
            feat = feat.view(-1, feat.shape[-1])
            feats.append(feat)

            has_label = "labels" in data
            if has_label:
                label = self.labels
                label = label.reshape(-1, label.shape[-1])
                labels.append(label)

        feats = torch.cat(feats)
        print(1, rank, feats.shape)
        if has_label:
            labels = torch.cat(labels)
            print(1, rank, labels.shape)
        dist.barrier()

        feats_list = [torch.zeros_like(feats) for _ in range(world_size)]
        dist.all_gather(feats_list, feats)
        feats = torch.cat(feats_list)
        print(2, rank, feats.shape)
        if has_label:
            labels_list = [torch.zeros_like(labels) for _ in range(world_size)]
            dist.all_gather(labels_list, labels)
            labels = torch.cat(labels_list)
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
