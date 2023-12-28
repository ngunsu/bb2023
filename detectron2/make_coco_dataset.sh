mkdir coco_dataset
mkdir coco_dataset/train
mkdir coco_dataset/valid
mkdir coco_dataset/test
mkdir coco_dataset/train/images
mkdir coco_dataset/valid/images
mkdir coco_dataset/test/images

train="$1/train"
val="$1/valid"
test="$1/test"

python3 -m yolo2coco.py --images_path "$train/images" --labels_path "$train/labels" --out coco_dataset/train
python3 -m yolo2coco.py --images_path "$val/images" --labels_path "$val/labels" --out coco_dataset/valid
python3 -m yolo2coco.py --images_path "$test/images" --labels_path "$test/labels" --out coco_dataset/test

cp "$train/images/"* coco_dataset/train/images
cp "$val/images/"* coco_dataset/valid/images
cp "$test/images/"* coco_dataset/test/images
