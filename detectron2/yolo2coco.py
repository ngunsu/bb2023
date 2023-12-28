import os
import click
from pathlib import Path
import cv2
import json
import shutil


def yolo_to_normal(x_center, y_center, box_width, box_height, width, height):
    x = x_center * width
    y = y_center * height
    w = box_width * width
    h = box_height * height

    x_min = x - (w / 2)
    y_min = y - (h / 2)
    x_max = x + (w / 2)
    y_max = y + (h / 2)

    return x_min, y_min, x_max, y_max

def normalize_yolo_coords(gt:str, im_width, im_height, include_score=False, as_int=True):

    data = gt.split(' ')

    cls = data[0]
    x = float(data[1])
    y = float(data[2])
    w = float(data[3])
    h = float(data[4])
    if include_score:
        score = float(data[5])

    x1, y1, x2, y2 = yolo_to_normal(x, y, w, h, im_width, im_height)

    if as_int:
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

    if include_score:
        return cls, x1, y1, x2, y2, score
    else:
        return cls, x1, y1, x2, y2
    

@click.command()
@click.option('--images_path', default='.', type=click.Path(exists=True))
@click.option('--labels_path', default='.', type=click.Path(exists=True))
@click.option('--as_results/--no_as_results', default=False)
@click.option('--norm_coords/--no_norm_coords', default=False)
@click.option('--out', default='.')
def cli(labels_path, images_path, as_results, norm_coords, out):
    
    if not os.path.isdir(out):
    	os.mkdir(out)

    coco_json = {}
    results_coco_json = []

    coco_json['info'] = {
        "year": "2021",
        "version": "1.0",
        "description": "Exported from FiftyOne",
        "contributor": "Voxel51",
        "url": "https://fiftyone.ai",
        "date_created": "2021-01-19T09:48:27"
    }

    coco_json['licenses'] = [
        {
          "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
          "id": 1,
          "name": "Attribution-NonCommercial-ShareAlike License"
        }
    ]

    coco_json['categories'] = [
        {
            "id": 0,
            "name": "Inmaduro",
            "supercategory": "Inmaduro"
        },
        {
            "id": 1,
            "name": "MMaduro",
            "supercategory": "MMaduro"
        },
        {
            "id": 2,
            "name": "Maduro",
            "supercategory": "Maduro"
        }
    ]

    coco_json['images'] = []
    coco_json['annotations'] = []

    images_list =os.listdir(images_path)
    print(images_list)
    im_c = 0
    ann_c = 0

    for im_p in images_list:
        
        im_path = os.path.join(images_path, im_p)
        lb_path = os.path.join(labels_path, im_p.replace(Path(im_p).suffix, ".txt"))

        im = cv2.imread(im_path, 1)


        coco_json['images'].append({
            "id": im_c,
            "license": 0,
            "file_name": Path(im_path).name,
            "height": im.shape[0],
            "width": im.shape[1],
            "date_captured": 'null'
        })

        if os.path.isfile(lb_path):
            with open(lb_path, 'r') as f:
                data = f.readlines()

                if len(data) == 0:
                    continue

                if data[-1] == '':
                    data.pop()

                for gt in data:
                    if as_results:
                        if not norm_coords:
                            cls, x1, y1, x2, y2, score = normalize_yolo_coords(gt, im.shape[1], im.shape[0], include_score=True, as_int=False)
                        else:
                            cls, x1, y1, x2, y2, score = gt.split(' ')
                            x1 = round(float(x1), 5) if round(float(x1), 5) >= 0 else 0
                            y1 = round(float(y1), 5) if round(float(y1), 5) >= 0 else 0
                            x2 = round(float(x2), 5) if round(float(x2), 5) >= 0 else 0
                            y2 = round(float(y2), 5) if round(float(y2), 5) >= 0 else 0

                            score = float(score)
                    else:
                        cls, x1, y1, x2, y2 = normalize_yolo_coords(gt, im.shape[1], im.shape[0], as_int=False)

                    x1 = round(x1, 5)
                    y1 = round(y1, 5)
                    x2 = round(x2, 5)
                    y2 = round(y2, 5)

                    coco_json['annotations'].append(
                        {
                            "id": ann_c,
                            "image_id": im_c,
                            "category_id": int(cls),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]],
                            "area": (x2 - x1) * (y2 - y1),
                            "iscrowd": 0
                        }
                    )

                    if as_results:
                        results_coco_json.append(
                        {
                            'image_id': im_c,
                            'category_id': int(cls),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]],
                            "score": score
                        })

                    ann_c += 1

                    #print(cls, x1, y1, x2, y2)

        im_c += 1

    if as_results:
        json_formatted = json.dumps(results_coco_json, indent=4)

        with open(os.path.join(out, 'pd_coco_dataset.json'), 'w') as f:
            f.write(json_formatted)

    else:
        json_formatted = json.dumps(coco_json, indent=4)

        with open(os.path.join(out, 'gt_coco_dataset.json'), 'w') as f:
            f.write(json_formatted)

cli()

