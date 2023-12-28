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


global glob_sum
glob_sum = 0

def test(data,
         weights=None,
         cfg=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=False,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         base_map=0.5,
         infer_mode=None,
         task='test',
         no_tide=False):

    y_true = []
    y_pred = []
    
    if batch_size > 1:
        exit("\nTesting must be done with batch_size=1 for accurate metrics calculation.\n")

    all_outputs = {}

    if infer_mode == 'mask-rcnn':
        cfg_path = cfg

        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
        cfg.MODEL.WEIGHTS = os.path.join(weights[0])  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres   # set a custom testing threshold
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = iou_thres   # set a custom testing threshold
        cfg.TEST.DETECTIONS_PER_IMAGE = 300

        predictor = DefaultPredictor(cfg)

    elif infer_mode == 'mask-rcnn-trt':
        from infer import TensorRTInfer
        from image_batcher import ImageBatcher
        
        if task == 'test':  folder = 'test'
        if task == 'train':  folder = 'train'
        if task == 'val':  folder = 'valid'

        images_folder = os.path.join(Path(data).parent, folder, 'images')

        trt_infer = TensorRTInfer(weights[0])
        batcher = ImageBatcher(images_folder, *trt_infer.input_spec(), config_file=cfg)

        all_dets = {}

        for batch, images, scales in batcher.get_batch():

            im_data = []

            detections = trt_infer.infer(batch, scales, conf_thres)
            
            for d in detections[0]:
                im_data.append([d['ymin'].item(), d['xmin'].item(), d['ymax'].item(), d['xmax'].item(), d['score'].item(), d['class']])

            all_dets[images[0]] = im_data
        
    elif infer_mode == 'yolov7-trt':
        trt_engine = YOLOv7_TRT_engine(weights[0], imgsz)

    elif infer_mode == 'rt-detr':
        from ultralytics.models import RTDETR

        model = RTDETR(weights[0])
        

    elif infer_mode == 'rt-detr-ts':

        conf_thres = conf_thres
        multi_label = False
        iou_thres=iou_thres
        classes=None
        agnostic=False
        labels=()

        ts_model = torch.jit.load(weights[0])

    device = 'cuda:0'
    training = False

    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model

        if infer_mode != 'yolov7':
            #model = attempt_load(weights, map_location=device)  # load FP32 model
            gs = 32 #max(int(model.stride.max()), 32)  # grid size (max stride)
            imgsz = check_img_size(imgsz, s=gs)  # check img_size
        else:
            model = attempt_load(weights, map_location=device)  # load FP32 model
            gs = max(int(model.stride.max()), 32)  # grid size (max stride)
            # print("GS", gs)
            imgsz = check_img_size(imgsz, s=gs)  # check img_size

            if trace:
                model = TracedModel(model, device, imgsz)

            half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
            if half:
                model.half()

            # Configure
            model.eval()


    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(base_map, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,  #<--------- ??????
                                       prefix=colorstr(f'{task}: '))[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    #names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    names = {0: 'Inmaduro', 1: 'MMaduro', 2: 'Maduro'}
    coco91class = coco80_to_coco91_class()
    print('\nDetection Metrics\n')
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'F1', f'mAP@{base_map}')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)

    #images_list = os.listdir(images_folder)

    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):

        if infer_mode == 'mask-rcnn-trt':
            
            out = all_dets[paths[0]]
            # for o in out: print(o)

        if infer_mode == 'yolov7':
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            #targets = targets.to(device)
            #nb, _, height, width = img.shape  # batch size, channels, height, width

        #im_path = os.path.join(images_folder, im_name)
        im = cv2.imread(paths[0], 1)

        targets = targets.to(device)
        nb, _, height, width = img.shape
        
        with torch.no_grad():
            if infer_mode != 'yolov7':
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)

            if infer_mode == 'yolov7':
                # Run model
                t = time_synchronized()
                out, train_out = model(img, augment=augment)  # inference and training outputs
                t0 += time_synchronized() - t

                # Compute loss
                if compute_loss:
                    loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

                # Run NMS
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
                lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
                t = time_synchronized()
                out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
                t1 += time_synchronized() - t
            
            if infer_mode == 'mask-rcnn':
                out = []
                outputs = predictor(im)
                outputs = outputs['instances'].to('cpu')

                boxes = outputs.pred_boxes if outputs.has("pred_boxes") else None
                classes = outputs.pred_classes.tolist() if outputs.has("pred_classes") else None
                scores = outputs.scores if outputs.has("scores") else None

                n_preds = 0
                for j, box in enumerate(boxes):
                    pred = [box[0].item(), box[1].item(), box[2].item(), box[3].item(), scores[j].item(), classes[j]]
                    out.append(pred)
                    # print(pred)

            if infer_mode == 'yolov7-trt':
                dets = trt_engine.predict(im, conf_thres)

                out = []
                
                for d in dets:
                    box = d[2:]
                    score = d[1]
                    cls = int(d[0])

                    out.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), score, cls])

                    # print([box[0], box[1], box[2], box[3], score, cls])

            if infer_mode == 'rt-detr-ts':
                im = cv2.imread(paths[0], 1)
                lb_image_tensor = util.preprocess(model, [im], half=True)

                out = []

                with torch.no_grad():
                    outputs = ts_model(lb_image_tensor)
                    preds = util.rtdetr_postprocess(outputs, im, conf_thres, classes)
                    
                    for pred in preds:
                        if not len(pred):
                            continue

                        x1 = int(pred[0][0].detach().cpu().item())
                        y1 = int(pred[0][1].detach().cpu().item())
                        x2 = int(pred[0][2].detach().cpu().item())
                        y2 = int(pred[0][3].detach().cpu().item())

                        prob = pred[0][4].detach().cpu().item()
                        cls = int(pred[0][5].detach().cpu().item())

                        out.append([x1, y1, x2, y2, prob, cls])

            if infer_mode == 'rt-detr':
                out = []

                dets = model.predict(paths[0], conf=0.25, iou=0.65, verbose=False)

                for det in dets[0].boxes.cpu().numpy():
                    out.append([det.xyxy[0][0], det.xyxy[0][1], det.xyxy[0][2], det.xyxy[0][3], det.conf, det.cls])


            if infer_mode != 'yolov7':            
                out = [torch.tensor(out).to(device)]
                im_name = Path(paths[0]).name
                all_outputs[im_name] = out

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            predn = pred.clone()
            if infer_mode == 'yolov7':
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
                im_name = Path(paths[0]).name
                all_outputs[im_name] = predn.unsqueeze(0)

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)

            if nl:
                # native Yolov7 impl
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]).to(device)
                #print(tbox)
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                #print(tbox)
                #if plots:
                #    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv   # iou_thres is 1xn # If IOU of pred is > 0.5 or 0.55 or 0.6 ... 0.95 adds the pred to the respective iou array
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            #print(stats)

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        # Modified to get map75 instead of 50
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.75, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    # pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    # print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    f1 = round((2 * mp * mr) / (mp + mr), 3)
    print(pf % ('all', seen, nt.sum(), mp, mr, f1, map50))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            name = names[c]
            # Just a translation
            if names[c] == 'Inmaduro':
                name = 'Unripe'
            if names[c] == 'MMaduro':
                name = 'Pint'
            if names[c] == 'Maduro':
                name = 'Ripe'
            # print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            f1 = round((2 * p[i] * r[i]) / (p[i] + r[i]), 3)
            print(pf % (name, seen, nt[c], p[i], r[i], f1, ap50[i]))
            
            
    if opt.task == 'test' and no_tide == False:
        #print(all_outputs)

        coco_json = []

        with open('coco_gt/gt_coco_dataset.json', 'r') as f:
            coco_gt = json.load(f)

        for data in coco_gt['images']:

            preds = all_outputs[data['file_name']]
            # print(preds)

            for pred in preds[0]:
                x1 = int(pred[0].detach().cpu().item())
                y1 = int(pred[1].detach().cpu().item())
                x2 = int(pred[2].detach().cpu().item())
                y2 = int(pred[3].detach().cpu().item())

                prob = pred[4].detach().cpu().item()
                cls = int(pred[5].detach().cpu().item())

                coco_json.append({
                'image_id': data['id'],
                'category_id': cls,
                'bbox':[
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                ],
                'segmentation':[ # TODO: FIX THIS
                    [
                        x1,
                        y1,
                        x2,
                        y1,
                        x2,
                        y2,
                        x1,
                        y2,
                        x1,
                        y1
                    ]
                ],
                'score': prob
            })

        with open(f'pd_coco_{infer_mode}.json', 'w') as f:
            json.dump(coco_json, f, indent=4)

        tide = TIDE()
        tide.evaluate(datasets.COCO('coco_gt/gt_coco_dataset.json'), datasets.COCOResult(f'pd_coco_{infer_mode}.json'), mode=TIDE.BOX, pos_threshold=0.75)
        res = tide.summarize()

        os.remove(f'pd_coco_{infer_mode}.json')

    # Print speeds
    #t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    t = 0
    #if not training:
    #    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    # model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='.cfg', help='Detectron2 cfg file.')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--base-map', type=float, default=0.75, help='The mAP to measure.')
    parser.add_argument('--infer-mode', type=str, default='yolov7', help='Type of model to process, yolov7, yolov7-trt, mask-rcnn, mask-rcnn-trt, rt-detr or rt-detr-ts.')
    parser.add_argument('--no-tide', action='store_true', help='Do not calculate the TIDE errors.')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.cfg,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             base_map=opt.base_map,
             infer_mode=opt.infer_mode,
             task=opt.task,
             no_tide=opt.no_tide
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
