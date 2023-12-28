import click
import os
import shutil
from tidecv import TIDE, datasets

@click.command()
@click.option('--gt_path', '-gt', type=click.Path(exists=True), help='Path to the ground truth COCO dataset')
@click.option('--pd_path', '-pd', type=click.Path(exists=True), help='Path to the predicted COCO dataset')
@click.option('--out', '-o', help='Path to the predicted COCO dataset')
def evaluate_object_detection(gt_path, pd_path, out):
    if os.path.isdir(out):
        shutil.rmtree(out, ignore_errors=True)
    os.mkdir(out)

    tide = TIDE()
    tide.evaluate(datasets.COCO(gt_path), datasets.COCOResult(pd_path), mode=TIDE.BOX, pos_threshold=0.75)
    res = tide.summarize()
    tide.plot(out)

    with open(os.path.join(out, 'results.txt'), 'w') as f:
        f.write(res)

if __name__ == '__main__':
    evaluate_object_detection()
