import click
from infer import TensorRTInfer
from image_batcher import ImageBatcher
import time
import numpy as np


@click.command()
@click.option('--warmup', default=100, help='Warmup value.')
@click.option('--times', default=100, help='Number of times to run.')
@click.option('--trt_model', default='', help='Path to the TRT model.')
@click.option('--cfg_path', default='', help='Path to the configuration file.')
def run(warmup, times, trt_model, cfg_path):

    images_folder = 'test_ims'

    trt_infer = TensorRTInfer(trt_model)
    batcher = ImageBatcher(images_folder, *trt_infer.input_spec(), config_file=cfg_path)

    all_dets = {}

    sample, image, scale = next(batcher.get_batch())

    measured_times = []
    for i in range(warmup + times):
        if i >= warmup:
            start = time.perf_counter()
            detections = trt_infer.infer(sample, scale, 0.25)
            end = time.perf_counter()

            measured_times.append(end - start)
        else:
            detections = trt_infer.infer(sample, scale, 0.25)

    im_data = []

    detections = trt_infer.infer(sample, scale, 0.25)
    
    for d in detections[0]:
        im_data.append([d['ymin'].item(), d['xmin'].item(), d['ymax'].item(), d['xmax'].item(), d['score'].item(), d['class']])
    
    print(im_data)

    print('Avg: ', np.mean(measured_times) * 1000)
    print('std: ', np.std(measured_times) * 1000)

if __name__ == '__main__':
    run()
