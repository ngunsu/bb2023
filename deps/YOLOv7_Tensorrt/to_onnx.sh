# Reparametrize models
cd /workspace/yolov7
python3 reparametrize.py --model_path /workspace/models/blueberry_yolov7/model_best.pt --model_type yolov7
python3 reparametrize.py --model_path /workspace/models/blueberry_yolov7w6/model_best.pt --model_type yolov7-w6

# Export to ONNX
cd /workspace/deps/YOLOv7_Tensorrt
python3 export_onnx.py --weights /workspace/models/blueberry_yolov7tiny/model_best.pt --img-size 640
python3 export_onnx.py --weights /workspace/models/blueberry_yolov7/model_best_rep.pt --img-size 640
python3 export_onnx.py --weights /workspace/models/blueberry_yolov7w6/model_best_rep.pt --img-size 1280
