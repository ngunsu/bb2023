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
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.engine.hooks import BestCheckpointer
import click
from LossEvalHook import LossEvalHook
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
import copy
import torch


def custom_mapper(dataset_dict):
    
    return dataset_dict


class MyTrainer(DefaultTrainer):

    def __init__(self, cfg, one_epoch):
        self.one_epoch = one_epoch
        self.step = 0
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        augs = [
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomRotation(angle=[-10, 10])
        ]
        return build_detection_train_loader(cfg)#, mapper=DatasetMapper(cfg, is_train=True, augmentations=augs))


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        coco_evaluator = COCOEvaluator(dataset_name, output_dir=output_folder, max_dets_per_image=300)
        
        return coco_evaluator
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.one_epoch, # Frequency of calculation - every 100 iterations here
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))

        hooks.append(BestCheckpointer(self.one_epoch, self.checkpointer, 'validation_loss', 'min'))

        return hooks


@click.command()
@click.option('--data', help='The person to greet.')
@click.option('--epochs', default=300, help='Number of greetings.')
@click.option('--batch_size', default=8, help='Number of greetings.')
@click.option('--lr', default=0.001, help='Number of greetings.')
@click.option('--out', default='', help='Number of greetings.')
def cli(data, epochs, batch_size, lr, out):
    register_coco_instances("my_dataset_train", {}, os.path.join(data, 'train', 'gt_coco_dataset.json'), os.path.join(data, 'train', 'images'))
    register_coco_instances("my_dataset_val", {}, os.path.join(data, 'valid', 'gt_coco_dataset.json'), os.path.join(data, 'valid', 'images'))
    register_coco_instances("my_dataset_test", {}, os.path.join(data, 'test', 'gt_coco_dataset.json'), os.path.join(data, 'test', 'images'))

    print("Dataset Registered.")

    one_epoch = int(len(os.listdir(f'{data}/train/images')) / batch_size)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = out
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.TEST.EVAL_PERIOD =  one_epoch
    cfg.SOLVER.MOMENTUM = 0.937
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = batch_size  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = epochs * one_epoch
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    #cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    #cfg.INPUT.MAX_SIZE_TRAIN = 640
    #cfg.INPUT.MIN_SIZE_TEST = 640
    #cfg.INPUT.MAX_SIZE_TEST = 640
    
    # Augs

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(cfg.OUTPUT_DIR, 'custom_cfg.yaml'), 'w') as f:
        f.write(cfg.dump())

    print(cfg.OUTPUT_DIR)

    trainer = MyTrainer(cfg, one_epoch)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode

    for i, d in enumerate(random.sample(os.listdir(f'{data}/test/images'), 3)):    
        im = cv2.imread(f'{data}/test/images/{d}',1)
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=None, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(f'{i}.jpg',  out.get_image()[:, :, ::-1])


if __name__ == '__main__':
    cli()
