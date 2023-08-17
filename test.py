import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)

import torch
import torch.backends.cudnn as cudnn
import time
from utils import *
from model import *
from data import *
import pickle
from config import CFG, resolution
import segmentation_models_pytorch as smp
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

cudnn.benchmark = True

@torch.no_grad()
def cls(data_type, batch_size, cnn, rnn=None, weights=None, seg_weights=None, contrast_learn=False):
    # Model
    seq_len = 1 if rnn is None else CFG.seq_len
    input_size = resolution[cnn]

    model = get_model(seq_len, cnn, rnn, weights, contrast_learn=contrast_learn, in_chans=1 if seg_weights is None else 4)
    model.to(CFG.device)
    model.eval()
    if seg_weights is not None:
        encoder = os.path.basename(seg_weights).split('_B')[0].replace('u-', '').replace('_', '-')
        seg_model = smp.Unet(encoder, in_channels=1, classes=3, activation='sigmoid')
        seg_model.load_state_dict(torch.load(seg_weights, map_location=CFG.device))
        seg_model.to(CFG.device)
        seg_model.eval()
        model.name = cnn+'-u-'+encoder.replace('-','_')

    # Dataloader
    dataset = ClsDataset(data_type, input_size, seq_len)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers,
                            pin_memory=True)

    print('-' * 30)
    print('Inference')
    
    true = torch.Tensor([])
    pred = torch.Tensor([]).to(CFG.device)

    since = time.time()

    for inputs, labels in dataloader:
        true = torch.cat((true, labels))
        inputs = inputs.to(CFG.device)
        B, L, C, H, W = inputs.shape
        inputs = inputs.view(-1, C, H, W)
        labels = labels.to(CFG.device)
        if seg_weights is not None:
            masks = seg_model(inputs)
            inputs = torch.cat((inputs, masks), dim=1)

        outputs = torch.sigmoid(model(inputs)).view(-1)
        pred = torch.cat((pred, outputs.detach()))

    true = np.array(true.cpu())
    pred = np.array(pred.cpu())

    time_elapsed = time.time() - since
    print(f'Inference complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    fps = len(dataloader.dataset) / time_elapsed
    print(f'FPS = {fps:.1f}')

    # Save results
    results = {'true': true, 'pred': pred, 'fps': fps}
    with open(os.path.join(CFG.PRED_DIR, data_type, f'{model.name}.pickle'), 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

@torch.no_grad()
def seg(weights):
    # Model
    batch_size = 32
    input_size = 224

    encoder = os.path.basename(weights).split('_')[0].replace('u-', '')
    model = smp.Unet(encoder, in_channels=1, classes=3, activation='sigmoid')
    model.load_state_dict(torch.load(weights, map_location=CFG.device))
    model.to(CFG.device)
    model.eval()

    # Dataloader
    dataset = SegDataset('val', input_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers,
                            pin_memory=True)
    print('-' * 30)
    print('Inference')

    since = time.time()

    true = np.array([])
    pred = np.array([])

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(CFG.device)
        B, L, C, H, W = inputs.shape
        inputs = inputs.view(-1, C, H, W)
        t = labels.detach().cpu().numpy()
        true = np.concatenate((true, t))
        outputs = model(inputs).detach().cpu().numpy()
        p = np.zeros(outputs.shape[0])
        for j in range(p.size):
            p[j] = np.array((outputs[j,0] > 0.5).any() and (outputs[j,1] > 0.5).any() and (outputs[j,2] > 0.5).any())
        pred = np.concatenate((pred, p))

    print(np.where(((true==0) * (pred==1))==1))
    print(accuracy_score(true, pred))
    p, r, f, _ = precision_recall_fscore_support(true, pred, average='binary')
    print(p, r, f)

    time_elapsed = time.time() - since
    print(f'Inference complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    fps = len(dataloader.dataset) / time_elapsed
    print(f'FPS = {fps:.1f}')


if __name__ == '__main__':
    pass
    #for data_type in ['test_cropped', 'GE', 'mindray', 'val']:
        # cls(data_type, 256, 'efficientnet_b0', rnn=None, weights='/home/ubuntu/workspace/@UGA/weights/efficientnet_b0_B256_E17_2022-09-22-10:04:24.pt', seg_weights=None, contrast_learn=False)
        # cls(data_type, 128, 'efficientnet_b1', rnn=None, weights='/home/ubuntu/workspace/@UGA/weights/efficientnet_b1_B128_E18_2022-09-22-10:24:37.pt', seg_weights=None, contrast_learn=False)
        # cls(data_type, 128, 'efficientnet_b2', rnn=None, weights='/home/ubuntu/workspace/@UGA/weights/efficientnet_b2_B128_E13_2022-09-22-10:56:10.pt', seg_weights=None, contrast_learn=False)

        # cls(data_type, 512, 'resnet18', rnn=None, weights='/home/ubuntu/workspace/@UGA/weights/resnet18_B512_E16_2022-09-22-09:08:34.pt', seg_weights=None, contrast_learn=False)
        # cls(data_type, 512, 'resnet34', rnn=None, weights='/home/ubuntu/workspace/@UGA/weights/resnet34_B512_E18_2022-09-22-09:29:58.pt', seg_weights=None, contrast_learn=False)
        # cls(data_type, 256, 'resnet50', rnn=None, weights='/home/ubuntu/workspace/@UGA/weights/resnet50_B256_E5_2022-09-22-09:50:16.pt', seg_weights=None, contrast_learn=False)
        # cls(data_type, 512, 'resnet18', rnn=None, weights='/home/ubuntu/workspace/@UGA/weights/resnet18_noAUG_B512_E7_2022-09-22-19:21:40.pt', seg_weights=None, contrast_learn=False)
        
        # cls(data_type, 256, 'resnet18', rnn='RNN', weights='/home/ubuntu/workspace/@UGA/weights/resnet18_RNN_B256_E19_2022-09-22-14:27:42.pt', seg_weights=None, contrast_learn=False)
        # cls(data_type, 256, 'resnet18', rnn='LSTM', weights='/home/ubuntu/workspace/@UGA/weights/resnet18_LSTM_B256_E10_2022-09-22-16:55:08.pt', seg_weights=None, contrast_learn=False)
        # cls(data_type, 256, 'resnet18', rnn='GRU', weights='/home/ubuntu/workspace/@UGA/weights/resnet18_GRU_B256_E4_2022-09-22-16:14:43.pt', seg_weights=None, contrast_learn=False)

        # cls(data_type, 64, 'resnet18', rnn=None, weights='/home/ubuntu/workspace/@UGA/weights/resnet18-u-resnet18_B512_E9_2022-09-22-19:42:49.pt', seg_weights='/home/ubuntu/workspace/@UGA/weights/u-resnet18_B64_E33_2022-09-22-18:45:36.pt', contrast_learn=False)
        # cls(data_type, 64, 'resnet18', rnn=None, weights='/home/ubuntu/workspace/@UGA/weights/resnet18-u-efficientnet_b2_B512_E4_2022-09-22-19:56:58.pt', seg_weights='/home/ubuntu/workspace/@UGA/weights/u-efficientnet-b2_B64_E45_2022-09-22-18:52:54.pt', contrast_learn=False)
        # cls(data_type, 256, 'resnet18', rnn=None, weights='/home/ubuntu/workspace/@UGA/weights/resnet18_SCL_B256_E19_2022-09-22-20:59:54.pt', seg_weights=None, contrast_learn=True)
