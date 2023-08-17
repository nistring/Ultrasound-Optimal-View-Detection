# Optimal View Detection for Ultrasound-guided Supraclavicular Block
https://doi.org/10.21203/rs.3.rs-2843354/v1
## Description

![image](https://github.com/nistring/Ultrasound-Optimal-View-Detection/assets/71208448/1a54927a-befb-4805-ae6e-08fe44cdd6b9)
This is a computer-aided diagnosis (CADx) system that can determine the optimal view for complete supraclavicular block(SCB) in real time. The segmentation network readily assigns three anatomically integral structures : brachial plexus(yellow), subclavian artery(red), and 1st rib(blue). The classification network estimates an optimallity score of the current image for practicing SCB. A score close to 100 simply means that the image is highly adjusted for SCB.

### Dependencies
  - python == 3.9
  - pytorch == 1.11.0
  - opencv-python == 4.5.5.64

For the rest, refer to requirements.txt

### Getting started

  ```
  git clone https://github.com/nistring/UGA.git
  pip install -r requirements.txt
  ```
Pretrained weights from https://drive.google.com/drive/folders/1is1dVDRL_owmRBxEQD5pQGGXRQxzXu6u?usp=share_link

First, connect an ultrasound machine to PC.

To execute the realtime optimal view detecting application, for example
  ```
  python realtime.py --cls_arch resnet34 --seg_arch resnet34 --cls_weights [path_to_cls_weights] --seg_weights [path_to_seg_weights]
  ```
Press R to manually assign region of interest(ROI).

Press Q to quit the program.

## Results

A demo example of realtime application. Automatically, bounding box is found and can visualize probability of optimal-view in SCB and grad-cam or segmentation predictions.

![00020481 mp4_20230711_095121](https://github.com/nistring/Ultrasound-Optimal-View-Detection/assets/71208448/f6ff103a-9d74-46d3-beb2-2f172b4bb83f)

Receiver operating characteristic curve and precision recall curve. A simple pretrained Resnet34 backbone is found to be effective.

![image](https://github.com/nistring/Ultrasound-Optimal-View-Detection/assets/71208448/11960fc2-fc05-4109-a357-0a3db275af3a)



