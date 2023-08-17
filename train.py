import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
import datetime
from torch.utils.data import DataLoader
from utils import *
from model import *
from data_loader import *
from config import *
import segmentation_models_pytorch as smp

cudnn.benchmark = True

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]," "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# Supervised Contrastive Learning
def train_SCL(cnn):
    batch_size = 256

    since = time.time()
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    criterion = SupConLoss()

    model = get_model(1, cnn, contrast_learn=True)

    for stage in ["contrast_learn", "classification"]:
        # Parameters for early stopping
        epoch_loss = dict.fromkeys(["train", "val"])
        best_loss = 100
        trigger_times = 0

        # Model
        if stage == "classification":
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 1)
        model.to(device)
        best_model = model
        best_epoch = 1
        optimizer = optim.Adam(model.parameters(), lr=cls_lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cls_step_size, gamma=gamma)

        while batch_size > 1:
            try:
                print("Batch size = ", batch_size)

                # Dataloader
                dataset = {}
                weight = {}
                dataloader = {}
                for phase in ["train", "val"]:
                    dataset[phase] = ClsDataset(
                        phase,
                        input_size,
                        contrast_learn=True if stage == "contrast_learn" else False,
                    )
                    weight[phase] = dataset[phase].pos_weight()
                    dataloader[phase] = DataLoader(
                        dataset[phase],
                        batch_size=batch_size,
                        shuffle=True if phase == "train" else False,
                        num_workers=num_workers,
                        pin_memory=True,
                    )

                visualize_augmentations(dataset["train"])

                learning = True

                for epoch in range(1, cls_max_epoch):
                    print("-" * 30)
                    print(f"Epoch {epoch}")

                    for phase in ["train", "val"]:
                        if phase == "train":
                            model.train()
                        else:
                            model.eval()

                        running_loss = 0.0

                        for i, (inputs, labels) in enumerate(dataloader[phase]):
                            bsz = labels.shape[0]
                            inputs = inputs.to(device)
                            B, L, C, H, W = inputs.shape
                            inputs = inputs.view(-1, C, H, W)
                            labels = labels.to(device)

                            optimizer.zero_grad()

                            with torch.set_grad_enabled(phase == "train"):
                                outputs = model(inputs)
                                if stage == "contrast_learn":
                                    outputs = nn.functional.normalize(outputs, dim=1)
                                    f1, f2 = torch.split(outputs, [bsz, bsz], dim=0)
                                    outputs = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                                else:
                                    outputs = outputs.view(-1)
                                    pos_weight = torch.ones_like(outputs, requires_grad=False, device=device) * weight[phase]
                                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                                loss = criterion(outputs, labels)

                                if phase == "train":
                                    loss.backward()
                                    optimizer.step()

                                batch_loss = loss.item()

                                print(f"[{i+1:3}/{len(dataloader[phase])}] : batch loss = {batch_loss:.4f}")
                                running_loss += batch_loss * bsz

                        epoch_loss[phase] = running_loss / len(dataset[phase])
                        print(f"{phase} loss = {epoch_loss[phase]:.4f}")

                        if phase == "train":
                            scheduler.step()

                        if phase == "val":
                            save_results(epoch, epoch_loss, model.name, run_date)
                            if epoch_loss["val"] < best_loss:
                                trigger_times = 0
                                best_model = model
                                best_epoch = epoch
                                best_loss = epoch_loss["val"]
                            else:
                                trigger_times += 1

                            if (trigger_times >= patience) and (epoch >= cls_max_epoch):
                                print("Early stopping!")
                                learning = False

                    if learning is False:
                        break
            except RuntimeError:
                # Automatically adjust batch size if exceeds memory size.
                batch_size = batch_size // 2
            else:
                print(f"\n{stage} ended.\n")
                break

    save_weights(best_model, model.name, run_date, best_epoch, batch_size, since)


def train_cls(cnn, rnn=None, cnn_weights=None, seg_weights=None):
    """Train classification model (optimal view detection)

    Args:
        cnn (str): Specify name of CNN model.
        rnn (str, optional): Specify name of RNN model if needed. Defaults to None.
        cnn_weights (_type_, optional): Defaults to None.
        seg_weights (_type_, optional): Defaults to None.
    """

    batch_size = 256
    seq_len = 1 if rnn is None else seq_len

    since = time.time()
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    # Parameters for early stopping
    epoch_loss = dict.fromkeys(["train", "val"])
    best_loss = 100
    trigger_times = 0

    # Model
    model = get_model(seq_len, cnn, rnn, cnn_weights, in_chans=1 if seg_weights is None else 4)
    model.to(device)
    best_model = model
    best_epoch = 1
    optimizer = optim.Adam(model.parameters(), lr=cls_lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cls_step_size, gamma=0.1)

    # Load segmentation model if indicated
    if seg_weights is not None:
        encoder = os.path.basename(seg_weights).split("_B")[0].replace("u-", "").replace("_", "-")
        seg_model = smp.Unet(encoder, in_channels=1, classes=3, activation="sigmoid")
        seg_model.load_state_dict(torch.load(seg_weights, map_location=device))
        seg_model.to(device)
        seg_model.eval()
        model.name = cnn + "-u-" + encoder.replace("-", "_")

    while batch_size > 1:
        try:
            print("Batch size = ", batch_size)

            # Dataloader
            dataset = {}
            weight = {}
            dataloader = {}

            for phase in ["train", "val"]:
                dataset[phase] = ClsDataset(phase, input_size, seq_len)
                weight[phase] = dataset[phase].pos_weight()
                dataloader[phase] = DataLoader(
                    dataset[phase],
                    batch_size=batch_size,
                    shuffle=True if phase == "train" else False,
                    num_workers=num_workers,
                    pin_memory=True,
                )

            visualize_augmentations(dataset["train"])

            for epoch in range(1, cls_max_epoch):
                print("-" * 30)
                print(f"Epoch {epoch}")

                for phase in ["train", "val"]:
                    if phase == "train":
                        model.train()
                    else:
                        model.eval()

                    running_loss = 0.0

                    for i, (inputs, labels) in enumerate(dataloader[phase]):
                        inputs = inputs.to(device)
                        B, L, C, H, W = inputs.shape
                        inputs = inputs.view(-1, C, H, W)
                        labels = labels.to(device)

                        optimizer.zero_grad()
                        if seg_weights is not None:
                            with torch.no_grad():
                                masks = seg_model(inputs)
                                inputs = torch.cat((inputs, masks), dim=1)

                        with torch.set_grad_enabled(phase == "train"):
                            outputs = model(inputs).view(-1)
                            pos_weight = torch.ones_like(outputs, requires_grad=False, device=device) * weight[phase]
                            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                            loss = criterion(outputs, labels)

                            if phase == "train":
                                loss.backward()
                                optimizer.step()

                            batch_loss = loss.item()

                            print(f"[{i+1:3}/{len(dataloader[phase])}] : batch loss = {batch_loss:.4f}")
                            running_loss += batch_loss * B

                    epoch_loss[phase] = running_loss / len(dataset[phase])
                    print(f"{phase} loss = {epoch_loss[phase]:.4f}")

                    if phase == "train":
                        scheduler.step()

                    if phase == "val":
                        save_results(epoch, epoch_loss, model.name, run_date)
                        if epoch_loss["val"] < best_loss:
                            trigger_times = 0
                            best_model = model
                            best_epoch = epoch
                            best_loss = epoch_loss["val"]
                        else:
                            trigger_times += 1

                        if (trigger_times >= patience) and (epoch >= cls_max_epoch):
                            print("Early stopping!")
                            save_weights(
                                best_model,
                                model.name,
                                run_date,
                                best_epoch,
                                batch_size,
                                since,
                            )
                            return

        except RuntimeError:
            batch_size = batch_size // 2
        else:
            print("Training ended")
            save_weights(best_model, model.name, run_date, best_epoch, batch_size, since)
            return


def train_seg(encoder):
    """# Train instance segmentation model (subclavian artery / brachial plexus / 1st rib)

    Args:
        encoder (str): Name of encoder(CNN) model
    """
    batch_size = 64
    input_size = 224

    since = time.time()
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    # Parameters for early stopping
    epoch_loss = dict.fromkeys(["train", "val"])
    best_loss = 100
    trigger_times = 0

    # Model
    model = smp.Unet(encoder, in_channels=1, classes=3, activation="sigmoid")
    model.to(device)
    best_model = model
    best_epoch = 1
    optimizer = optim.Adam(model.parameters(), lr=seg_lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=seg_step_size, gamma=gamma)
    criterion = smp.losses.DiceLoss("multilabel", from_logits=False)

    while batch_size > 1:
        try:
            print("Batch size = ", batch_size)

            # Dataloader
            dataset = {}
            dataloader = {}

            for phase in ["train", "val"]:
                dataset[phase] = SegDataset(phase, input_size)
                dataloader[phase] = DataLoader(
                    dataset[phase],
                    batch_size=batch_size,
                    shuffle=True if phase == "train" else False,
                    num_workers=num_workers,
                    pin_memory=True,
                )

            visualize_augmentations(dataset["train"])

            for epoch in range(1, seg_max_epoch):
                print("-" * 30)
                print(f"Epoch {epoch}")

                for phase in ["train", "val"]:
                    if phase == "train":
                        model.train()
                    else:
                        model.eval()

                    running_loss = 0.0

                    for i, (inputs, masks) in enumerate(dataloader[phase]):
                        inputs = inputs.to(device)
                        masks = torch.stack(masks, dim=1).type(torch.float32)
                        masks = masks.to(device)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == "train"):
                            outputs = model(inputs)

                            loss = criterion(outputs, masks)

                            if phase == "train":
                                loss.backward()
                                optimizer.step()

                            batch_loss = loss.item()

                            print(f"[{i+1:3}/{len(dataloader[phase])}] : batch loss = {batch_loss:.4f}")
                            running_loss += batch_loss * inputs.size(0)

                    epoch_loss[phase] = running_loss / len(dataset[phase])
                    print(f"{phase} loss = {epoch_loss[phase]:.4f}")

                    if phase == "train":
                        scheduler.step()

                    if phase == "val":
                        save_results(epoch, epoch_loss, model.name, run_date)
                        if epoch_loss["val"] < best_loss:
                            trigger_times = 0
                            best_model = model
                            best_epoch = epoch
                            best_loss = epoch_loss["val"]
                        else:
                            trigger_times += 1

                        if (trigger_times >= patience) and (epoch >= seg_max_epoch):
                            print("Early stopping!")
                            save_weights(
                                best_model,
                                model.name,
                                run_date,
                                best_epoch,
                                batch_size,
                                since,
                            )
                            return

        except RuntimeError:
            batch_size = batch_size // 2
        else:
            print("Training ended")
            save_weights(best_model, model.name, run_date, best_epoch, batch_size, since)
            return


if __name__ == "__main__":
    pass
    # train_cls('resnet18')
    # train_cls('resnet34')
    # train_cls('resnet50')
    # train_cls('resnet101')
    # train_cls('resnet152')

    # train_seg('resnet18')
    # train_seg('resnet34')
    # train_seg('resnet50')
    # train_seg('resnet101')
    # train_seg('resnet152')

    # train_cls('resnet34', 'RNN', cnn_weights='/home/ubuntu/workspace/@UGA/weights/resnet34_B512_E8_2022-10-13-21:43:09.pt')
    # train_cls('resnet34', 'LSTM', cnn_weights='/home/ubuntu/workspace/@UGA/weights/resnet34_B512_E8_2022-10-13-21:43:09.pt')
    # train_cls('resnet34', 'GRU', cnn_weights='/home/ubuntu/workspace/@UGA/weights/resnet34_B512_E8_2022-10-13-21:43:09.pt')
    # train_cls('resnet34', seg_weights='/home/ubuntu/workspace/@UGA/weights/u-resnet34_B64_E58_2022-10-14-07:32:13.pt')
    # train_SCL('resnet34')
    # train_cls('resnet34') # noAUG
