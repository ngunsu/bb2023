python3 export.py --weights /workspace/models/blueberry_rt-detr-l/model_best.pt
mv /workspace/models/blueberry_rt-detr-l/model_best.torchscript /workspace/models/blueberry_rt-detr-l/model_best_fp32.torchscript
python3 export.py --weights /workspace/models/blueberry_rt-detr-l/model_best.pt --half
mv /workspace/models/blueberry_rt-detr-l/model_best.torchscript /workspace/models/blueberry_rt-detr-l/model_best_fp16.torchscript
