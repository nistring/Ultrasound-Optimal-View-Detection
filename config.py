import torch
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))

num_workers=16

cls_lr=0.0001
cls_step_size=10
cls_max_epoch=30

seg_lr=0.001
seg_step_size=25
seg_max_epoch=100

gamma=0.1
sampling_rate=0.2 # one sample per 0.5 sec # 0.2 sec in practice
buffer_size=1
seq_len = 8
patience = 3

thresh_binary = 1 # midray 1
kernel_size = 8 # mindray 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLS_DATA_DIR = cur_dir+'/img/data/cls'
SEG_DATA_DIR = cur_dir+'/img/data/seg'
IMG_EVAL_DIR = cur_dir+'/img/evaluation'
VIDEO_EVAL_DIR = cur_dir+'/video/evaluation'
WEIGHT_DIR = cur_dir+'/weights'
FIG_DIR = cur_dir+'/figures'
VIDEO_DIR = cur_dir+'/video/data'
VIDEO_APP_DIR = cur_dir+'/video/applied'

objects = ['artery','nerve','rib']
obj_color = {'artery':[0,0,255], 'nerve':[0,255,255], 'rib':[255,0,0]}

input_size = 224