# RT-DETR

It is recommended to delete all the cache files from the dataset before training. Both Ultralytics and YOLOv7 will create **labels.cache** files when training and testing. If you train a YOLOv7 model and then attempt to train an Ultralytics model, Ultralytics will try to read the labels.cache file created by YOLOv7 and fail to load it.

```
rm /workspace/datasets/blueberry_dataset/train/labels.cache
rm /workspace/datasets/blueberry_dataset/valid/labels.cache
rm /workspace/datasets/blueberry_dataset/test/labels.cache
```

To simply train the RT-DETR model as specified in the paper you can simply run this script:

```
python3 train.py --data /workspace/datasets/blueberry_dataset/data.yaml
```

To learn the full extent on how to train a model using the Ultralytics API it's recomended to follow the official instructions and tutorials.

- [Offical Repo.](https://github.com/ultralytics/ultralytics/)

## Export model

The model can be exported to torchscript via the export.py script.

To convert to a fp32 torchscript representation:

```
python3 export.py --weights /workspace/models/blueberry_rt-detr-l/model_best.pt
```

To convert to a fp16 torchscript representation:

```
python3 export.py --weights /workspace/models/blueberry_rt-detr-l/model_best.pt --half
```

The resulting model is saved into the same folder where the model specified in **--weights** resides.


To test the exported model you can run the test.py script in /workspace.

```
cd /workspace
python3 test.py --weights /workspace/models/blueberry_rt-detr-l/model_best.torchscript --data /workspace/datasets/blueberry_dataset/data.yaml --batch-size 1 --img-size 640 --conf-thres 0.25 --iou-thres 0.65 --task test --device 0 --base-map 0.75 --infer-mode rt-detr-ts --no-tide
cd /workspace/rt-detr
```
