# Some basic setup:
# Setup detectron2 logger
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
import click
import json


def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        bbox1 (list or numpy array): The first bounding box in [xmin, ymin, xmax, ymax] format.
        bbox2 (list or numpy array): The second bounding box in [xmin, ymin, xmax, ymax] format.
    
    Returns:
        The IoU of the two bounding boxes as a floating point number.
    """
    # calculate the area of each bounding box
    area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    # calculate the intersection of the bounding boxes
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # calculate the union of the bounding boxes
    union_area = area1 + area2 - inter_area

    # calculate IoU
    iou = inter_area / union_area

    return iou


def main(eval_data_path, cfg_path, model_path):
    register_coco_instances("my_dataset_test", {}, os.path.join(eval_data_path, "annotations.json"), os.path.join(eval_data_path, "images"))

    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.WEIGHTS = os.path.join(model_path)  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.65   # set a custom testing threshold
    cfg.TEST.DETECTIONS_PER_IMAGE = 300

    predictor = DefaultPredictor(cfg)
    
    with open(os.path.join(eval_data_path, "annotations.json")) as f:
        json_data = json.load(f)

    coco_json = []


    metrics = {}

    for c in json_data['categories']:
        metrics[c['id']] = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': len([x for x in json_data['annotations'] if x['category_id'] == c['id']])
        }
        
    for im_data in json_data['images']:
        print(im_data["file_name"], im_data['id'])
        im = cv2.imread(os.path.join(eval_data_path, 'images', im_data["file_name"]), 1)
        outputs = predictor(im)
        outputs = outputs['instances'].to('cpu')

        boxes = outputs.pred_boxes if outputs.has("pred_boxes") else None
        classes = outputs.pred_classes.tolist() if outputs.has("pred_classes") else None
        scores = outputs.scores if outputs.has("scores") else None

        anns = [x for x in json_data['annotations'] if x['image_id'] == im_data['id']]

        id_matches = []

        # bboxes in the gt left to match
        
        for i, bbox in enumerate(boxes):
            pred_box = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            f_pred_box = [round(bbox[0].item(), 2), round(bbox[1].item(), 2), round(bbox[2].item(), 2), round(bbox[3].item(), 2)]
            
            for ann in anns:
                gt_box = [int(ann['bbox'][0]), int(ann['bbox'][1]), int(ann['bbox'][0] + ann['bbox'][2]), int(ann['bbox'][1] + ann['bbox'][3])]

                iou = calculate_iou(pred_box, gt_box)
                
                if iou >= 0.75:
                    if ann['category_id'] == classes[i]:
                        metrics[ann['category_id']]['true_positives'] += 1

                        # check bboxes that matched the anns in the gt
                        id_matches.append(i)

            #print(pred_box, scores[i].item(), classes[i])
            coco_json.append({
                'image_id': im_data['id'],
                'category_id': classes[i],
                'bbox':[
                    f_pred_box[0],
                    f_pred_box[1],
                    f_pred_box[2] - f_pred_box[0],
                    f_pred_box[3] - f_pred_box[1]
                ],
                'segmentation':[ # TODO: FIX THIS
                    [
                        f_pred_box[0],
                        f_pred_box[1],
                        f_pred_box[2],
                        f_pred_box[1],
                        f_pred_box[2],
                        f_pred_box[3],
                        f_pred_box[0],
                        f_pred_box[3],
                        f_pred_box[0],
                        f_pred_box[1]
                    ]
                ],
                'score': scores[i].item()
            })

        non_matched = []

        for i, bbox in enumerate(boxes):
            if i in id_matches:
                continue
            else:
                non_matched.append(i)

        # the non matched are false positives
        for i in range(len(non_matched)):
            metrics[classes[i]]['false_positives'] += 1
    
    with open('pd_coco_0.01_2.json', 'w') as f:
        json.dump(coco_json, f, indent=4)


    for i in metrics:
        metrics[i]['false_negatives'] =  metrics[i]['false_negatives'] - metrics[i]['true_positives']

    for i in metrics:
        presicion = metrics[i]['true_positives'] / (metrics[i]['true_positives'] + metrics[i]['false_positives'])
        recall = metrics[i]['true_positives'] / (metrics[i]['true_positives'] + metrics[i]['false_negatives'])
        print(f'Clase: {i}, Precision: {presicion}, Recall: {recall}')

    evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/", max_dets_per_image=300)
    val_loader = build_detection_test_loader(cfg, "my_dataset_test", mapper=None)
    inference_on_dataset(predictor.model, val_loader, evaluator)

    from detectron2.utils.visualizer import ColorMode

    for i, d in enumerate(random.sample(os.listdir(os.path.join(eval_data_path, 'images')), 3)):    
        im = cv2.imread(os.path.join(eval_data_path, 'images', d),1)
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=None, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(f'{i}.jpg',  out.get_image()[:, :, ::-1])


@click.command()
@click.option('--eval_data', help='The person to greet.')
@click.option('--cfg_file', default='', help='Number of greetings.')
@click.option('--model_path', default='', help='Number of greetings.')
def cli(eval_data, cfg_file, model_path):
    main(eval_data, cfg_file, model_path)

if __name__ == '__main__':
    cli()
