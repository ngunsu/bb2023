# Comprehensive Analysis of Model Errors in Blueberry Detection and Maturity Classification: Identifying Limitations and Proposing Future Improvements in Agricultural Monitoring.

Repository for the code used in the paper

```tex
@Article{agriculture14010018,
AUTHOR = {Aguilera, Cristhian A. and Figueroa-Flores, Carola and Aguilera, Cristhian and Navarrete, Cesar},
TITLE = {Comprehensive Analysis of Model Errors in Blueberry Detection and Maturity Classification: Identifying Limitations and Proposing Future Improvements in Agricultural Monitoring},
JOURNAL = {Agriculture},
VOLUME = {14},
YEAR = {2024},
NUMBER = {1},
ARTICLE-NUMBER = {18},
URL = {https://www.mdpi.com/2077-0472/14/1/18},
ISSN = {2077-0472},
ABSTRACT = {In blueberry farming, accurately assessing maturity is critical to efficient harvesting. Deep Learning solutions, which are increasingly popular in this area, often undergo evaluation through metrics like mean average precision (mAP). However, these metrics may only partially capture the actual performance of the models, especially in settings with limited resources like those in agricultural drones or robots. To address this, our study evaluates Deep Learning models, such as YOLOv7, RT-DETR, and Mask-RCNN, for detecting and classifying blueberries. We perform these evaluations on both powerful computers and embedded systems. Using Type-Influence Detector Error (TIDE) analysis, we closely examine the accuracy of these models. Our research reveals that partial occlusions commonly cause errors, and optimizing these models for embedded devices can increase their speed without losing precision. This work improves the understanding of object detection models for blueberry detection and maturity estimation.},
DOI = {10.3390/agriculture14010018}
}
```

## Requirements

- Docker
- nvidia-docker

Build the docker container.

```bash
export WORKSPACE=/absolute_path_to_this_folder
docker build -t blueberry_detection -f docker/Dockerfile .
```

## Replicate results

The first step is to launch the docker container. **All commands are run inside the container**.

```bash
# shell 1
sh docker/run.sh
# shell 2
docker exec -it blueberry_detection /bin/bash
```

The second step is to download the dataset & models which will be evaluated.

```bash
sh download_dataset.sh
sh download_models.sh
```

### Object Detection Metrics

For each architecture, 3 models were trained and their metrics averaged. The detection metrics reported are:

| Model          | Class                         | Precision                        | Recall                           | F1                               | mAP75                            |
|----------------|-------------------------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|
| YOLOv7-tiny    | Ripe<br>Pint<br>Unripe<br>All | 0.547<br>0.568<br>0.485<br>0.533 | 0.387<br>0.433<br>0.323<br>0.380 | 0.443<br>0.489<br>0.388<br>0.443 | 0.330<br>0.364<br>0.231<br>0.309 |
| YOLOv7-default | Ripe<br>Pint<br>Unripe<br>All | 0.626<br>0.641<br>0.605<br>0.624 | 0.456<br>0.508<br>0.415<br>0.46  | 0.528<br>0.567<br>0.492<br>0.530 | 0.435<br>0.432<br>0.348<br>0.405 |
| YOLOv7-w6      | Ripe<br>Pint<br>Unripe<br>All | 0.598<br>0.631<br>0.591<br>0.607 | 0.500<br>0.494<br>0.457<br>0.484 | 0.544<br>0.554<br>0.516<br>0.539 | 0.445<br>0.431<br>0.381<br>0.419 |
| RT-DETR-L      | Ripe<br>Pint<br>Unripe<br>All | 0.606<br>0.627<br>0.544<br>0.592 | 0.429<br>0.329<br>0.380<br>0.380 | 0.502<br>0.431<br>0.447<br>0.462 | 0.393<br>0.282<br>0.289<br>0.321 |
| Mask-RCNN      | Ripe<br>Pint<br>Unripe<br>All | 0.612<br>0.680<br>0.582<br>0.625 | 0.490<br>0.574<br>0.488<br>0.518 | 0.543<br>0.622<br>0.530<br>0.565 | 0.447<br>0.558<br>0.426<br>0.477 |


You can reproduce the results by running the following scripts.

#### YOLOv7-tiny

```bash
python3 test.py --weights /workspace/models/blueberry_yolov7tiny/model1.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode yolov7 --no-tide
python3 test.py --weights /workspace/models/blueberry_yolov7tiny/model_best.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode yolov7 --no-tide
python3 test.py --weights /workspace/models/blueberry_yolov7tiny/model3.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode yolov7 --no-tide
```

#### YOLOv7-default

```bash
python3 test.py --weights /workspace/models/blueberry_yolov7/model_best.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode yolov7 --no-tide
python3 test.py --weights /workspace/models/blueberry_yolov7/model2.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode yolov7 --no-tide
python3 test.py --weights /workspace/models/blueberry_yolov7/model3.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode yolov7 --no-tide
```

#### YOLOv7-w6

```bash
python3 test.py --weights /workspace/models/blueberry_yolov7w6/model1.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 1280 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode yolov7 --no-tide
python3 test.py --weights /workspace/models/blueberry_yolov7w6/model2.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 1280 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode yolov7 --no-tide
python3 test.py --weights /workspace/models/blueberry_yolov7w6/model_best.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 1280 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode yolov7 --no-tide
```

#### RT-DETR-L

```bash
python3 test.py --weights /workspace/models/blueberry_rt-detr-l/model_best.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode rt-detr --no-tide
python3 test.py --weights /workspace/models/blueberry_rt-detr-l/model2.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode rt-detr --no-tide
python3 test.py --weights /workspace/models/blueberry_rt-detr-l/model3.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode rt-detr --no-tide
```

#### Mask-RCNN

```bash
python3 test.py --weights /workspace/models/blueberry_maskrcnn/model_best.pth --cfg /workspace/models/blueberry_maskrcnn/custom_cfg.yaml --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode mask-rcnn --no-tide
python3 test.py --weights /workspace/models/blueberry_maskrcnn/model2.pth --cfg /workspace/models/blueberry_maskrcnn/custom_cfg.yaml --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode mask-rcnn --no-tide
python3 test.py --weights /workspace/models/blueberry_maskrcnn/model3.pth --cfg /workspace/models/blueberry_maskrcnn/custom_cfg.yaml --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode mask-rcnn --no-tide
```

### TIDE Errors

TIDE requires the results in COCO format to calculate errors, so it is essential to convert the results from the YOLO format into a format compliant with COCO standards

The following script will perform the conversion automatically.

```bash
cd detectron2
python3 yolo2coco.py --images_path /workspace/datasets/blueberry_dataset/test/images/ --labels_path /workspace/datasets/blueberry_dataset/test/labels --out ../coco_gt
cd /workspace
```


The errors metrics reported by each model are:

| Model       | Cls  | Loc   | Both | Dupe | Bkg  | Miss  | FP    | FN    |
|-------------|------|-------|------|------|------|-------|-------|-------|
| YOLOv7-tiny | 5.73 | 32.3  | 0.31 | 0.0  | 0.27 | 10.87 | 10.35 | 43.14 |
| YOLOv7-default| 5.8  | 28.41 | 0.21 | 0.0  | 0.36 | 10.29 | 10.25 | 38.28 |
| YOLOv7-w6   | 6.19 | 26.47 | 0.26 | 0.0  | 0.27 | 11.42 | 9.47  | 38.95 |
| RT-DETR-L   | 8.43 | 35.52 | 0.41 | 0.0  | 0.71 | 6.71  | 11.88 | 40.75 |
| Mask-RCNN   | 3.26 | 30.47 | 0.17 | 0.0  | 0.77 | 8.56  | 10.93 | 34.35 |


You can measure the TIDE errors & mAP of each model with the following:

```
python3 test.py --weights /workspace/models/blueberry_yolov7tiny/model_best.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode yolov7
python3 test.py --weights /workspace/models/blueberry_yolov7/model_best.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode yolov7
python3 test.py --weights /workspace/models/blueberry_yolov7w6/model_best.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 1280 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode yolov7
python3 test.py --weights /workspace/models/blueberry_rt-detr-l/model_best.pt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode rt-detr
python3 test.py --weights /workspace/models/blueberry_maskrcnn/model_best.pth --cfg /workspace/models/blueberry_maskrcnn/custom_cfg.yaml --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode mask-rcnn
```

### Runtime Benchmark

The average time it took each model to do inference over an image. Each model does inference 100 times as warmup, as to not pollute the final results, then each model does another 100 inferences which are recorded. Finally we get the mean of the 100 times recorded and then we calculate the standard deviation.

#### GTX 3080

Times recorded on a computer system with a 12th Gen Core i7 CPU, 32GB of RAM, a 1TB SSD, and an NVIDIA RTX3080TI 10GB GPU.

| Model          | fp32   | std   | fp16   | std   |
|----------------|--------|-------|--------|-------|
| YOLOv7-tiny    | 3.308  | 0.013 | 2.252  | 0.049 |
| YOLOv7-default | 8.059  | 0.046 | 3.823  | 0.059 |
| YOLOv7-w6      | 19.551 | 0.142 | 7.677  | 0.081 |
| RT-DETR-L      | 11.512 | 0.385 | 8.933  | 0.522 |
| Mask-RCNN      | 34.301 | 0.755 | 16.998 | 0.206 |

For this step you need to convert the models to a TensorRT engine.

To convert the YOLOv7 models:
```
cd deps/YOLOv7_Tensorrt
sh to_onnx.sh
sh to_trt.sh
cd /workspace
```

To convert the Mask-RCNN model:

```
cd detectron2
sh to_trt.sh
cd /workspace
```

The RT-DETR-L model cannot be converted to a TRT engine, therefore it's coverted to a torchscript model.

```
cd rt-detr
sh to_torchscript.sh
cd /workspace
```

FP16 inference times.

```
python3 benchmark.py --weights /workspace/models/blueberry_yolov7tiny/model_best_fp16.trt --imgsz 640 --infer-type yolov7-trt --im_folder /workspace/datasets/blueberry_dataset/test/images/
python3 benchmark.py --weights /workspace/models/blueberry_yolov7/model_best_fp16.trt --imgsz 640 --infer-type yolov7-trt --im_folder /workspace/datasets/blueberry_dataset/test/images/
python3 benchmark.py --weights /workspace/models/blueberry_yolov7w6/model_best_fp16.trt --imgsz 1280 --infer-type yolov7-trt --im_folder /workspace/datasets/blueberry_dataset/test/images/
python3 benchmark.py --weights /workspace/models/blueberry_rt-detr-l/model_best_fp16.torchscript --infer-type rt-detr-ts --im_folder /workspace/datasets/blueberry_dataset/test/images/ --half
python3 benchmark.py --weights /workspace/models/blueberry_maskrcnn/model_best_fp16.trt --infer-type mask-rcnn-trt --im_folder /workspace/datasets/blueberry_dataset/test/images/
```

FP32 inference times.

```
python3 benchmark.py --weights /workspace/models/blueberry_yolov7tiny/model_best_fp32.trt --imgsz 640 --infer-type yolov7-trt --im_folder /workspace/datasets/blueberry_dataset/test/images/
python3 benchmark.py --weights /workspace/models/blueberry_yolov7/model_best_fp32.trt --imgsz 640 --infer-type yolov7-trt --im_folder /workspace/datasets/blueberry_dataset/test/images/
python3 benchmark.py --weights /workspace/models/blueberry_yolov7w6/model_best_fp32.trt --imgsz 1280 --infer-type yolov7-trt --im_folder /workspace/datasets/blueberry_dataset/test/images/
python3 benchmark.py --weights /workspace/models/blueberry_rt-detr-l/model_best_fp32.torchscript --infer-type rt-detr-ts --im_folder /workspace/datasets/blueberry_dataset/test/images/
python3 benchmark.py --weights /workspace/models/blueberry_maskrcnn/model_best_fp32.trt --infer-type mask-rcnn-trt --im_folder /workspace/datasets/blueberry_dataset/test/images/
```


## Training

You can train models by following the instructions in each folder corresponding to its architecture.

- [YOLOv7](yolov7)
- [RT-DETR](rt-detr)
- [Mask-RCNN](detectron2)
