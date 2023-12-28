#FP16 version
trtexec --onnx=/workspace/models/blueberry_yolov7tiny/model_best.onnx --saveEngine=/workspace/models/blueberry_yolov7tiny/model_best_fp16.trt --workspace=10240 --fp16
trtexec --onnx=/workspace/models/blueberry_yolov7/model_best_rep.onnx --saveEngine=/workspace/models/blueberry_yolov7/model_best_fp16.trt --workspace=10240 --fp16
trtexec --onnx=/workspace/models/blueberry_yolov7w6/model_best_rep.onnx --saveEngine=/workspace/models/blueberry_yolov7w6/model_best_fp16.trt --workspace=10240 --fp16

#FP32 version
trtexec --onnx=/workspace/models/blueberry_yolov7tiny/model_best.onnx --saveEngine=/workspace/models/blueberry_yolov7tiny/model_best_fp32.trt --workspace=10240
trtexec --onnx=/workspace/models/blueberry_yolov7/model_best_rep.onnx --saveEngine=/workspace/models/blueberry_yolov7/model_best_fp32.trt --workspace=10240
trtexec --onnx=/workspace/models/blueberry_yolov7w6/model_best_rep.onnx --saveEngine=/workspace/models/blueberry_yolov7w6/model_best_fp32.trt --workspace=10240
