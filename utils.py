import torch
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
from config import *
import time
from pytorch_grad_cam import GradCAM

cmap = plt.get_cmap("plasma")
test_list = ["GE", "mindray", "test_cropped"]


def crop(frame, roi=None):
    """Crops foreground ultrasound image.

    Args:
        frame (np.ndarray): Raw ultrasound image.
        roi (tuple, optional): Coordinates of proposed bounding box. Defaults to None.

    Returns:
        np.ndarray: Cropped image.
        float: Mean.
        float: Standard deviation.
        tuple: Coordinates of bounding box.
    """

    original_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if roi is not None:
        # Ignore outside of ROI
        x, y, w, h = roi
        roi = original_img[y : y + h, x : x + w]
        img = np.zeros_like(original_img)
        # Make background pixels be 0 so that they are seperated easily from foreground
        img[y : y + h, x : x + w] = cv2.threshold(
            roi, thresh_binary, 255, cv2.THRESH_TOZERO
        )[1]
    else:
        img = cv2.threshold(original_img, thresh_binary, 255, cv2.THRESH_TOZERO)[1]

    ret = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # Find a contour with maximum area
    contours, hierarchy = cv2.findContours(
        ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    cropped_img = original_img[y : y + h, x : x + w]

    # Calculate mean & std for normalization
    mean, std = cv2.meanStdDev(cropped_img)
    return cropped_img, mean, std, (x, y, w, h)


def save_results(epoch, epoch_loss, arch, run_date):
    result = "epoch: {:d}, train_loss : {:.4f}, val_loss : {:.4f}".format(
        epoch, epoch_loss["train"], epoch_loss["val"]
    )
    if not os.path.exists(VIDEO_EVAL_DIR):
        os.makedirs(VIDEO_EVAL_DIR)
    fp = open(os.path.join(VIDEO_EVAL_DIR, f"{arch}_{run_date}.txt"), "a")
    fp.write(result + "\n")
    fp.close()


def save_weights(model, arch, run_date, best_epoch, batch_size, since):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    filename = f"{arch}_B{batch_size}_E{best_epoch}_{run_date}.pt"
    path = os.path.join(WEIGHT_DIR, filename)
    torch.save(model.state_dict(), path)

    # Record training time
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Saving weights to {filename}")

    return path


def load_unmet_state_dict(model, path):
    """Loads pretrained model weights even if the layers of model are subtly modified.

    Args:
        model (nn.Module):
        path (str):
    """
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def show_cam_on_image(img, cam, bbox, pred):
    """Overlays grad-cam on given image.
    Grad-cam results are multiplied by predicted optimal-view probability.

    Args:
        img (np.ndarray): Original image.
        cam (np.ndarray): Grad-cam.
        bbox (tuple): Coordinates of bounding box.
        pred (float): Predicted optimal-view probability.

    Returns:
        np.ndarray: Overlayed image.
    """
    x, y, w, h = bbox
    x = int(x + w * 0.027)
    y = int(y + h * 0.027)
    w = int(w * 0.973)
    h = int(h * 0.973)
    img = img.astype(np.float32)

    colormap = cv2.COLORMAP_JET
    cam = cv2.resize(cam, (w, h)) * pred
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    image_weight = np.stack([cam, cam, cam], axis=2) / 2
    img[y : y + h, x : x + w] = (
        img[y : y + h, x : x + w] * (1 - image_weight) + image_weight * heatmap
    )

    return np.uint8(img)


def show_seg_on_image(img, seg, bbox, pred):
    """Overlays segmentation results on input image.
    Segmentation results are multiplied by predicted optimal-view probability.

    Args:
        img (np.ndarray): Original image.
        seg (np.ndarray): Segmentation results.
        bbox (tuple): Coordinates of bounding box.
        pred (float): Predicted optimal-view probability.

    Returns:
        np.ndarray: Overlayed image.
    """
    x, y, w, h = bbox
    x = int(x + w * 0.027)
    y = int(y + h * 0.027)
    w = int(w * 0.973)
    h = int(h * 0.973)
    img = img.astype(np.float32)

    for i, obj in enumerate(objects):
        mask = cv2.resize(seg[i], (w, h)) * pred
        color_mapping = np.full((h, w, 3), obj_color[obj])
        image_weight = np.stack([mask, mask, mask], axis=2) / 2
        img[y : y + h, x : x + w] = (
            img[y : y + h, x : x + w] * (1 - image_weight)
            + image_weight * color_mapping
        )

    return np.uint8(img)


def draw_status(frame, bbox, pred, latency=None):
    """Draws bounding box, probability of optimal-view, and latency.

    Args:
        frame (np.ndarray): Frame to be displayed.
        bbox (tuple): Coordinates of bounding box.
        pred (float): Predicted optimal-view probability.
        latency (float, optional): Latency(realtime applications). Defaults to None.

    Returns:
        _type_: _description_
    """
    x, y, w, h = bbox
    off_x, off_y = 5, 50
    rect_color = tuple([int(c * 255) for c in cmap(pred)[:3]])
    score = int(pred * 100)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Bounding box
    cv2.rectangle(
        frame, (23 + off_x, 48 + off_y), (77 + off_x, 152 + off_y), (255, 255, 255), 2
    )
    cv2.rectangle(
        frame,
        (25 + off_x, 149 - score + off_y),
        (75 + off_x, 150 + off_y),
        rect_color,
        -1,
    )
    cv2.putText(
        frame,
        f"score = {score}",
        (25 + off_x, 200 + off_y),
        cv2.FONT_HERSHEY_COMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    if latency != None:
        cv2.putText(
            frame,
            f"latency = {latency:.3f}",
            (25 + off_x, 250 + off_y),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    return frame


def set_cam(model):
    """Sets Grad-cam with specified target layers.

    Args:
        model (nn.Module):

    Raises:
        ValueError: Raises error when given unsupported models for GradCam

    Returns:
        GradCAM:
    """
    if "resnet" in model.name:
        target_layers = [model.layer4[-1].conv2]
    elif "efficientnet" in model.name:
        target_layers = [model.conv_head]
    else:
        raise ValueError
    return GradCAM(
        model=model,
        target_layers=target_layers,
        use_cuda=True if torch.cuda.is_available() else False,
    )


def get_optimal_num():
    for data_type in ["train", "val", "test_cropped", "GE", "mindray"]:
        true = np.load(os.path.join(CLS_DATA_DIR, data_type + "_label.npy"))
        total = true.size
        optimal = np.sum(true)
        non_optimal = total - optimal
        print(data_type)
        print(f"total : {total}")
        print(f"optimal : {optimal}")
        print(f"non_optimal : {non_optimal}")
