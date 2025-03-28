import os
import datetime
import cv2
from PIL import Image
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms 

from models.cnn_lstm import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
								transforms.Resize([224, 224]),
							    transforms.ToTensor(),
							    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
								])
transform_color = transforms.Compose([
								transforms.Resize([224, 224]),
							    transforms.ToTensor(),
							    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
								])  

# color_class = ['colorless', 'green', 'red', 'white', 'yellow']
color_class = ['无色', '绿色', '红色', '白色', '黄色']
lstm_class = ['5s单闪', '单闪', '双闪', '顿光', '间歇闪', '定光', '快闪', '莫尔斯D', '莫尔斯P', '莫尔斯X', '三闪']

def create_nml_model():
    # 模型
    color_weights = "color_epoch.pth"
    cnn_weights = "cnn_epoch.pth"
    rnn_weights = "rnn_epoch.pth"

    # colorless:0    green:1     red:2    white:3
    model_color = colorClassification(num_classes=4)
    model_color.load_state_dict(torch.load(color_weights, map_location='cpu'))
    model_color.to(device).eval()

    model_cnn = ResCNNEncoder(fc_hidden1=2048, fc_hidden2=768, drop_p=0.0, CNN_embed_dim=512).to(device)
    model_cnn.load_state_dict(torch.load(cnn_weights,  map_location='cpu'))
    model_cnn.to(device).eval()

    model_rnn = DecoderRNN(CNN_embed_dim=512, h_RNN_layers=3, h_RNN=512,
                        h_FC_dim=256, drop_p=0.0, num_classes=11).to(device)
    model_rnn.load_state_dict(torch.load(rnn_weights, map_location='cpu'))
    model_rnn.to(device).eval()

    return [model_color, model_cnn, model_rnn]


def nmlcls(model_list, video_path):
    """
    推理航标灯质识别的流程
    """
    model_color, model_cnn, model_rnn = model_list

    print(video_path)
    cap = cv2.VideoCapture(video_path)
    count = cap.get(7)
    fps = 20
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    list_img_v = []
    list_img_color = []
    c = 1
    color_label = [0, 0, 0, 0]
    while c <= count:
        rval, frame = cap.read()
        if c % fps == 1:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ## 转成hsv图片
            hsv = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2HSV)  # hsv
            v = hsv[:, :, 2]  # 亮度图
            v = Image.fromarray(v)  # 将numpy转成PIL才可以转成tensor

            frame = transform_color(frame)
            frame = frame.float()
            v = transform(v)
            v = v.float()
            list_img_v.append(v) # 存储v通道的列表进行时序预测

            # 对每一个图片帧进行颜色分类
            frame = frame.unsqueeze(0).to(device)
            color_yuce = model_color(frame)
            color_yuce = int((color_yuce.cpu().argmax()).numpy())
            color_label[color_yuce] = color_label[color_yuce] + 1
        c += 1
    num1 = color_label.index(max(color_label[1:-1]))
    color_label[num1] = 0
    num2 = color_label.index(max(color_label[1:-1]))

    res_color = ""
    if color_class[num1] or color_class[num2] == 'colorless':
        print("单色")
        print("为", color_class[num1], "色。", end=";")
        res_color = color_class[num1]
    else:
        print("双色")
        print("先", color_class[num1],",后", color_class[num2], "色。", end=";")
        res_color = color_class[num1]

    # 时序数据处理
    list_img_v = torch.stack(list_img_v, dim=0)
    list_img_v  = list_img_v.view(1, len(list_img_v), 1, 224, 224).to(device)

    # 模型预测
    hh = model_cnn(list_img_v)
    output = model_rnn(hh).view(11)
    lstm_yuce = int((output.cpu().argmax()).numpy())

    return [{"color": res_color, "flash": lstm_class[lstm_yuce]}]