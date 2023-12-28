# import
from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import select_device, is_parallel
import yaml
from pathlib import Path
import click


def reparametrize(model_path, model_type):
    
    device = select_device('0', batch_size=1)
    # model trained by cfg/training/*.yaml
    ckpt = torch.load(model_path, map_location=device)
    nc = ckpt['model'].nc
    # reparameterized model in cfg/deploy/*.yaml

    # if model_type == 'yolov7-tiny':
    #    cfg_file = 'cfg/deploy/yolov7-tiny.yaml'
    if model_type == 'yolov7':
        cfg_file = 'cfg/deploy/yolov7.yaml'
    elif model_type == 'yolov7x':
        cfg_file = 'cfg/deploy/yolov7x.yaml'
    elif model_type == 'yolov7-w6':
        cfg_file = 'cfg/deploy/yolov7-w6.yaml'
    elif model_type == 'yolov7-d6':
        cfg_file = 'cfg/deploy/yolov7-d6.yaml'
    elif model_type == 'yolov7-e6':
        cfg_file = 'cfg/deploy/yolov7-e6.yaml'
    elif model_type == 'yolov7-e6e':
        cfg_file = 'cfg/deploy/yolov7-e6e.yaml'
    else:
        raise NotImplementedError()

    model = Model(cfg_file, ch=3, nc=nc).to(device)

    with open(cfg_file) as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    anchors = len(yml['anchors'][0]) // 2
    
    if model_type == 'yolov7-tiny':
        # copy intersect weights
        state_dict = ckpt['model'].float().state_dict()
        exclude = []
        intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(intersect_state_dict, strict=False)
        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc

        # reparametrized YOLOR, 255=(80 + 5) * 3,  80 is coco number class.
        for i in range((model.nc+5)*anchors):       #calculate from above line
            model.state_dict()['model.77.m.0.weight'].data[i, :, :, :] *= state_dict['model.77.im.0.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.77.m.1.weight'].data[i, :, :, :] *= state_dict['model.77.im.1.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.77.m.2.weight'].data[i, :, :, :] *= state_dict['model.77.im.2.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.77.m.0.bias'].data += state_dict['model.77.m.0.weight'].mul(state_dict['model.77.ia.0.implicit']).sum(1).squeeze()
        model.state_dict()['model.77.m.1.bias'].data += state_dict['model.77.m.1.weight'].mul(state_dict['model.77.ia.1.implicit']).sum(1).squeeze()
        model.state_dict()['model.77.m.2.bias'].data += state_dict['model.77.m.2.weight'].mul(state_dict['model.77.ia.2.implicit']).sum(1).squeeze()
        model.state_dict()['model.77.m.0.bias'].data *= state_dict['model.77.im.0.implicit'].data.squeeze()
        model.state_dict()['model.77.m.1.bias'].data *= state_dict['model.77.im.1.implicit'].data.squeeze()
        model.state_dict()['model.77.m.2.bias'].data *= state_dict['model.77.im.2.implicit'].data.squeeze()

    if model_type == 'yolov7':
        # copy intersect weights
        state_dict = ckpt['model'].float().state_dict()
        exclude = []
        intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(intersect_state_dict, strict=False)
        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc

        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
            model.state_dict()['model.105.m.0.weight'].data[i, :, :, :] *= state_dict['model.105.im.0.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.105.m.1.weight'].data[i, :, :, :] *= state_dict['model.105.im.1.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.105.m.2.weight'].data[i, :, :, :] *= state_dict['model.105.im.2.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.105.m.0.bias'].data += state_dict['model.105.m.0.weight'].mul(state_dict['model.105.ia.0.implicit']).sum(1).squeeze()
        model.state_dict()['model.105.m.1.bias'].data += state_dict['model.105.m.1.weight'].mul(state_dict['model.105.ia.1.implicit']).sum(1).squeeze()
        model.state_dict()['model.105.m.2.bias'].data += state_dict['model.105.m.2.weight'].mul(state_dict['model.105.ia.2.implicit']).sum(1).squeeze()
        model.state_dict()['model.105.m.0.bias'].data *= state_dict['model.105.im.0.implicit'].data.squeeze()
        model.state_dict()['model.105.m.1.bias'].data *= state_dict['model.105.im.1.implicit'].data.squeeze()
        model.state_dict()['model.105.m.2.bias'].data *= state_dict['model.105.im.2.implicit'].data.squeeze()

    elif model_type == 'yolov7x':
        # copy intersect weights
        state_dict = ckpt['model'].float().state_dict()
        exclude = []
        intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(intersect_state_dict, strict=False)
        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc

        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
            model.state_dict()['model.121.m.0.weight'].data[i, :, :, :] *= state_dict['model.121.im.0.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.121.m.1.weight'].data[i, :, :, :] *= state_dict['model.121.im.1.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.121.m.2.weight'].data[i, :, :, :] *= state_dict['model.121.im.2.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.121.m.0.bias'].data += state_dict['model.121.m.0.weight'].mul(state_dict['model.121.ia.0.implicit']).sum(1).squeeze()
        model.state_dict()['model.121.m.1.bias'].data += state_dict['model.121.m.1.weight'].mul(state_dict['model.121.ia.1.implicit']).sum(1).squeeze()
        model.state_dict()['model.121.m.2.bias'].data += state_dict['model.121.m.2.weight'].mul(state_dict['model.121.ia.2.implicit']).sum(1).squeeze()
        model.state_dict()['model.121.m.0.bias'].data *= state_dict['model.121.im.0.implicit'].data.squeeze()
        model.state_dict()['model.121.m.1.bias'].data *= state_dict['model.121.im.1.implicit'].data.squeeze()
        model.state_dict()['model.121.m.2.bias'].data *= state_dict['model.121.im.2.implicit'].data.squeeze()

    elif model_type == 'yolov7-w6':
        # copy intersect weights
        state_dict = ckpt['model'].float().state_dict()
        exclude = []
        intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(intersect_state_dict, strict=False)
        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc

        idx = 118
        idx2 = 122

        # copy weights of lead head
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data -= model.state_dict()['model.{}.m.0.weight'.format(idx)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data -= model.state_dict()['model.{}.m.1.weight'.format(idx)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data -= model.state_dict()['model.{}.m.2.weight'.format(idx)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data -= model.state_dict()['model.{}.m.3.weight'.format(idx)].data
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data -= model.state_dict()['model.{}.m.0.bias'.format(idx)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data -= model.state_dict()['model.{}.m.1.bias'.format(idx)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data -= model.state_dict()['model.{}.m.2.bias'.format(idx)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data -= model.state_dict()['model.{}.m.3.bias'.format(idx)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.bias'.format(idx2)].data

        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()

    elif model_type == 'yolov7-e6':
        # copy intersect weights
        state_dict = ckpt['model'].float().state_dict()
        exclude = []
        intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(intersect_state_dict, strict=False)
        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc

        idx = 140
        idx2 = 144

        # copy weights of lead head
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data -= model.state_dict()['model.{}.m.0.weight'.format(idx)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data -= model.state_dict()['model.{}.m.1.weight'.format(idx)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data -= model.state_dict()['model.{}.m.2.weight'.format(idx)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data -= model.state_dict()['model.{}.m.3.weight'.format(idx)].data
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data -= model.state_dict()['model.{}.m.0.bias'.format(idx)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data -= model.state_dict()['model.{}.m.1.bias'.format(idx)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data -= model.state_dict()['model.{}.m.2.bias'.format(idx)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data -= model.state_dict()['model.{}.m.3.bias'.format(idx)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.bias'.format(idx2)].data

        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()

    elif model_type == 'yolov7-d6':
        # copy intersect weights
        state_dict = ckpt['model'].float().state_dict()
        exclude = []
        intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(intersect_state_dict, strict=False)
        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc

        idx = 162
        idx2 = 166

        # copy weights of lead head
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data -= model.state_dict()['model.{}.m.0.weight'.format(idx)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data -= model.state_dict()['model.{}.m.1.weight'.format(idx)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data -= model.state_dict()['model.{}.m.2.weight'.format(idx)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data -= model.state_dict()['model.{}.m.3.weight'.format(idx)].data
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data -= model.state_dict()['model.{}.m.0.bias'.format(idx)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data -= model.state_dict()['model.{}.m.1.bias'.format(idx)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data -= model.state_dict()['model.{}.m.2.bias'.format(idx)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data -= model.state_dict()['model.{}.m.3.bias'.format(idx)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.bias'.format(idx2)].data

        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()

    elif model_type == 'yolov7-e6e':
        # copy intersect weights
        state_dict = ckpt['model'].float().state_dict()
        exclude = []
        intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(intersect_state_dict, strict=False)
        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc

        idx = 261
        idx2 = 265

        # copy weights of lead head
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data -= model.state_dict()['model.{}.m.0.weight'.format(idx)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data -= model.state_dict()['model.{}.m.1.weight'.format(idx)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data -= model.state_dict()['model.{}.m.2.weight'.format(idx)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data -= model.state_dict()['model.{}.m.3.weight'.format(idx)].data
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data -= model.state_dict()['model.{}.m.0.bias'.format(idx)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data -= model.state_dict()['model.{}.m.1.bias'.format(idx)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data -= model.state_dict()['model.{}.m.2.bias'.format(idx)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data -= model.state_dict()['model.{}.m.3.bias'.format(idx)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.bias'.format(idx2)].data

        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()



    # model to be saved
    ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
            'optimizer': None,
            'training_results': None,
            'epoch': -1}

    path = Path(model_path)
    name = path.stem
    suff = path.suffix
    parent_dir = path.parent.absolute()

    torch.save(ckpt, f'{parent_dir}/{name}_rep{suff}')


@click.command()
@click.option('--model_path', default='.', type=click.Path(exists=True))
@click.option('--model_type', type=str)
def cli(model_path, model_type):
    reparametrize(model_path, model_type)

if __name__ == '__main__':
    cli()
