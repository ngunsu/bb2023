from ultralytics import RTDETR
import click

@click.command()
@click.option('--weights', help='Number of greetings.')
@click.option('--half', is_flag=True, default=False, help='Sample image.')
@click.option('--device', type=int, default=0, help='Cuda device. eg: 0')
def cli(weights, half, device):
# Load a model
    model = RTDETR(weights)  # load a custom trained

    # Export the model
    model.export(format='torchscript', half=half, device=device)

if __name__ == '__main__':
    cli()
