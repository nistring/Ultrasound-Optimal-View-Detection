import torch
import time
import torch.backends.cudnn as cudnn
import cv2
from utils import *
from model import *
from data_loader import *
from multiprocessing import Queue
import threading
from config import *
import argparse
import segmentation_models_pytorch as smp

cudnn.benchmark = True

# Ultrasound probe -> [frame_que] -> model -> [result_que] -> display
frame_que = Queue(buffer_size)
result_que = Queue(buffer_size)
lock = threading.Lock()
delay = 1e-3

mask = None


def realtime_app(args):
    """On the model side, gets frame from frame_que then processes it and puts in result_que.

    Args:
        args ():
    """

    global mask

    pred = 0

    # Model
    model = get_model(1, args.cls_arch, weights=args.cls_weights)
    model.to(device)
    model.eval()

    seg_model = smp.Unet(args.seg_arch, in_channels=1, classes=3, activation="sigmoid")
    seg_model.load_state_dict(torch.load(args.seg_weights, map_location=device))
    seg_model.to(device)
    seg_model.eval()

    transform = get_transform("test", input_size, 1)

    while True:
        # Gets raw frame from frame_que.
        success = False
        while not success:
            time.sleep(delay)
            lock.acquire()
            if not frame_que.empty():
                frame = frame_que.get()
                success = True
            lock.release()
        since = time.time()

        # Preprocessing
        frame, mean, std, bbox = crop(frame, roi=mask)
        frame = transform(image=frame)["image"].unsqueeze(0)
        input = frame.to(device)

        # Inference
        p = torch.sigmoid(model(input)).cpu().item()
        seg = seg_model(input).squeeze(0).detach().cpu().numpy()

        pred = 0.5 * (p + pred)  # A very simple loss pass filter
        latency = time.time() - since

        # Puts back to result_que
        success = False
        while not success:
            lock.acquire()
            if not result_que.full():
                result_que.put((bbox, pred, seg, latency))
                success = True
                lock.release()
                break
            lock.release()
            time.sleep(delay)


def display():
    """Delivers unprocessed frames then retrieves processed ones and display them."""
    global mask
    # Receive frames from capture board
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_sampled = fps * sampling_rate
    idx = 0
    pred = 0.0
    bbox = (0, 0, 0, 0)
    latency = 0.0
    seg = None
    cv2.namedWindow("frame")

    while True:
        _, frame = cap.read()

        # Puts frame to frame_que
        lock.acquire()
        if idx >= frame_sampled:
            idx -= frame_sampled
            if not frame_que.full():
                frame_que.put(frame)

        if not result_que.empty():
            bbox, pred, seg, latency = result_que.get()
        lock.release()

        # Drawing
        frame = show_seg_on_image(frame, seg, bbox, pred)
        frame = draw_status(frame, bbox, pred, latency)

        # Display
        cv2.imshow("frame", frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord("q"):
            break
        if k & 0xFF == ord("r"):
            x, y, w, h = cv2.selectROI("frame", frame, False)
            if w and h:
                mask = [x, y, w, h]
        idx += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_model", "-cm", choices=["resnet18", "resnet34", "resnet50"], default="resnet34")
    parser.add_argument("--seg_model", "-sm", choices=["resnet18", "resnet34", "resnet50"], default="resnet34")
    parser.add_argument("--cls_weights", "-cw", required=True)
    parser.add_argument("--seg_weights", "-sw", required=True)
    args = parser.parse_args()

    t = threading.Thread(target=realtime_app, args=(args,))
    t.start()
    display()
    t.join()


if __name__ == "__main__":
    main()
