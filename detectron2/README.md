
# Detectron2 Module

## Usage

First you'll need to export the blueberry dataset you previously downloaded. Detectron2 requires the dataset to be in COCO format.
The resulting dataset is stored in the **coco_dataset** folder.

```
sh make_coco_dataset.sh /datasets/blueberry_dataset/
```

To train you can run the train.py script.

```
python3 train.py --data coco_dataset/ --epochs 300 --batch_size 2 --lr 0.01 --out trained_model
```

To test the metrics you will need to use the test.py script thats on the /workspace folder

```
cd /workspace
python3 test.py --weights detectron2/trained_model/model_best.pth --cfg detectron2/trained_model/custom_cfg.yaml --batch-size 1 --task test --data /datasets/blueberry_dataset/data.yaml --infer-mode mask-rcnn --base-map 0.75
```

## Export model

To export to a TensorRT engine.

```
python3 tools/deploy/export_onnx.py --sample-image 1344x1344.jpg --config-file /workspace/models/blueberry_maskrcnn/custom_cfg.yaml --export-method tracing --format onnx --output exported_models MODEL.WEIGHTS /workspace/models/blueberry_maskrcnn/model_best.pth MODEL.DEVICE cuda
create_onnx.py --exported_onnx exported_models/model.onnx --onnx exported_models/converted.onnx --det2_config /workspace/models/blueberry_maskrcnn/custom_cfg.yaml --det2_weights /workspace/models/blueberry_maskrcnn/model_best.pth --sample_image 1344x1344.jpg
python3 build_engine.py --onnx /workspace/detectron2/exported_models/converted.onnx --engine exported_models/model_best_fp16.trt --precision fp16
```

You can test the exported model via the test.py script in the /workspace folder.

```
cd /workspace
python3 test.py --weights /workspace/detectron2/exported_models/model_best_fp16.trt --cfg /workspace/models/blueberry_maskrcnn/custom_cfg.yaml --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode mask-rcnn-trt
cd detectron2
```

