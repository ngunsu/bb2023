# Training

To train a custom model it's recommended to read the official instructions given below.

To train the models as specified in the paper, you first need to download the COCO pre-trained models.

```
sh download_models.sh
```

It is recommended to delete all the cache files from the dataset before training. Both Ultralytics and YOLOv7 will create **labels.cache** files when training and testing. If you train an Ultralytics model and then attempt to train a YOLOv7 model, YOLOv7 will try to read the labels.cache file created by Ultralytics and fail to load it.

```
rm /workspace/datasets/blueberry_dataset/train/labels.cache
rm /workspace/datasets/blueberry_dataset/valid/labels.cache
rm /workspace/datasets/blueberry_dataset/test/labels.cache
```

To train the yolov7-tiny model.

```
python3 train.py --weights yolov7-tiny.pt --hyp data/hyp.scratch.custom_0.01.yaml --epochs 300 --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 8 --img-size 640
```

To train the yolov7 model.
```
python3 train.py --weights yolov7_training.pt --hyp data/hyp.scratch.custom_0.001.yaml --epochs 300 --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 8 --img-size 640
```

To train the yolov7-w6 model.
```
python3 train_aux.py --weights yolov7-w6_training.pt --hyp data/hyp.scratch.custom_0.01.yaml --epochs 300 --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 8 --img-size 1280
```

## Export model

To export to TensorRT, it is recommended to first reparametrize the model for faster inference. Yolov7-tiny cannot be reparametrized.

```
python3 reparametrize.py --model_path /workspace/models/blueberry_yolov7/model_best.pt --model_type yolov7
```

Then you can simply export to ONNX and then to TensorRT.

```
cd /workspace/deps/YOLOv7_Tensorrt
python3 export_onnx.py --weights /workspace/models/blueberry_yolov7/model_best_rep.pt --img-size 640
trtexec --onnx=/workspace/models/blueberry_yolov7/model_best_rep.onnx --saveEngine=/workspace/models/blueberry_yolov7/model_best_fp16.trt --workspace=10240 --fp16
cd /workspace/yolov7
```

To test the model you can run the test.py script in /workspace.

```
cd /workspace
python3 test.py --weights /workspace/models/blueberry_yolov7/model_best_fp16.trt --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode yolov7-trt --no-tide
cd /workspace/yolov7
```


