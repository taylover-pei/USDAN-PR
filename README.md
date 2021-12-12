# USDAN 
The implementation of [**Unified unsupervised and semi-supervised domain adaptation network for cross-scenario face anti-spoofing**](https://www.sciencedirect.com/science/article/abs/pii/S0031320321000753), which is accepted by Pattern Recognition 2021.

An overview of the proposed USDAN method:

<div align=center>
<img src="https://github.com/taylover-pei/USDAN-PR/blob/main/article/Architecture.png" width="700" height="345" />
</div>

## Congifuration Environment
- python 3.6 
- pytorch 0.4 
- torchvision 0.2
- cuda 8.0

## Pre-training

**Dataset.** 

Download the CASIA-FASD, Idiap Replay-Attack, and MSU-MFSD datasets.

**Data Pre-processing.** 

[MTCNN algorithm](https://ieeexplore.ieee.org/abstract/document/7553523) is utilized for face detection and face alignment. All the detected faces are normalized to 224$\times$224$\times$3, where only RGB channels are utilized for training. The exact codes that we used can be found [here](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection).

Put the processed frames in the path `$root/processed_data`

To be specific, we first utilize the MTCNN algorithm to process every frame of each video. And then, we utilize the `get_files` function in the `utils/utils.py` to sample frames during training. Finally, the information of selected frames are saved to the `choose_*.json` file.

**Data Label Generation.** 

Move to the `$root/USDAN_*/msu_casia/data_label/` and generate the data label list:
```python
python generate_label.py
```

## Training

Move to the folder `$root/USDAN_*/msu_casia/` and just run like this:
```python
python train_USDAN_*.py
```

The file `config.py` contains the hype-parameters used during training.

## Testing

Run like this:
```python
python da_test.py
```

## Citation
Please cite our paper if the code is helpful to your researches.
```
@InProceedings{Jia_2021_PR_USDAN,
    author = {Yunpei Jia and Jie Zhang and Shiguang Shan and Xilin Chen},
    title = {Unified Unsupervised and Semi-supervised Domain Adaptation Network for Cross-scenario Face Anti-spoofing},
    booktitle = {Pattern Recognition},
    year = {2021}
}
```




