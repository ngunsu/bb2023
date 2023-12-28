import argparse
import sys
sys.path.append("/workspace/yolov7")
sys.path.append("/workspace/detectron2")
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from tidecv import TIDE, datasets

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel
import sys
from collections import namedtuple
from torchvision import models, transforms
import cv2
from PIL import Image
from pathlib import Path
from yolo_engine import YOLOv7_TRT_engine
import util
from ultralytics import YOLO
import time
import click


@click.command()
@click.option('--weights', help='path to the weights.')
@click.option('--cfg', help='Cfg file for Detectron2 models.')
@click.option('--im_folder', help='Folder to process, only the first image is processed.')
@click.option('--imgsz', default=640, help='Size to resize the image to. Only necesary for YOLOv7 models.')
@click.option('--warmup', default=100, help='Number of Warmup inferences.')
@click.option('--times', default=100, help='Number of inferences to measure the average inference time.')
@click.option('--conf', default=0.25, help='Conf threshold.')
@click.option('--iou', default=0.65, help='NMS threashold.')
@click.option('--infer-type', default='yolov7', help='Type of model to process, yolov7-trt, mask-rcnn-trt or rt-detr-ts.')
@click.option('--half', is_flag=True, default=False, help='Convert inputs to half, you only need to specify this if you use rt-detr-ts.')
def cli(weights, cfg, im_folder, imgsz, warmup, times, conf, iou, infer_type, half):

    execs = []

    if infer_type == 'yolov7-trt':
        im = cv2.imread(os.path.join(im_folder, os.listdir(im_folder)[0]), 1)
        trt_engine = YOLOv7_TRT_engine(weights, imgsz)

        execs = trt_engine.benchmark(warmup, times, im)

    if infer_type == 'mask-rcnn-trt':
        from infer import TensorRTInfer
        from image_batcher import ImageBatcher

        images_folder = os.path.join(im_folder)

        trt_infer = TensorRTInfer(weights)
        batcher = ImageBatcher(images_folder, *trt_infer.input_spec(), config_file=cfg)

        for batch, images, scales in batcher.get_batch():

            for i in range(0, warmup + times):
                start = time.perf_counter()
                dets = trt_infer.infer(batch, scales, conf)
                end = time.perf_counter()

                if i > warmup -1:
                    execs.append(end - start)

            break
            

    if infer_type == 'rt-detr-ts':

        conf_thres = conf
        multi_label = False
        iou_thres=0.65
        classes=None
        agnostic=False
        labels=()

        model = torch.jit.load(weights)
        im = cv2.imread(os.path.join(im_folder, os.listdir(im_folder)[0]), 1)
        lb_image_tensor = util.preprocess(model, [im], half)

        for i in range(0, warmup + times):
            
            with torch.no_grad():
                start = time.perf_counter()
                outputs = model(lb_image_tensor)
                end = time.perf_counter()

            if i > warmup - 1:
                execs.append(end - start)


    print('MEAN:    ', round(np.mean(execs) * 1000, 3), 'ms.')
    print('STD:     ', round(np.std(execs) * 1000, 3), 'ms.')




def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocess_im(im, imgsz, stride, device, half):

    im = letterbox(im, imgsz, stride)[0]

    # Convert
    im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255.0  # 0 - 255 to 0.0 - 1.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    return im

if __name__ == '__main__':
    cli()
