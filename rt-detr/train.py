from ultralytics import RTDETR
import click


@click.command()
@click.option('--data', default=None, help='Path to the data yaml.')
def train(data):
    model = RTDETR("rtdetr-l.pt")
    model.info()  # display model information
    model.train(
    data=data,
    optimizer="SGD",
    amp=True,
    epochs=300,
    patience=100,
    batch=8,
    val=True,
    augment=True,
    lr0=0.001,
    lrf=0.1,
    mosaic=1,
    close_mosaic=0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.1,
    translate=0.2,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    )   # train
    
    
if __name__ == '__main__':
    train()
