import torch
from fl.model_factory import ModelFactory

model_params = torch.load('./cache/model/cifar10_model.pth')

model_json = {
    "model_type":
    "multi_classification",
    "learning_rate":
    0.001,
    "optimizer":
    "adam",
    "criterion":
    "cross_entropy",
    "layers": [{
        "type": "conv2d",
        "in_channels": 3,  # CIFAR-10 has 3 channels
        "out_channels": 32,
        "kernel_size": 3,
        "padding": 1,
        "stride": 1
    }, {
        "type": "relu"
    }, {
        "type": "maxpool",
        "kernel_size": 2,
        "stride": 2,
        "padding": 0
    }, {
        "type": "conv2d",
        "in_channels": 32,
        "out_channels": 64,
        "kernel_size": 3,
        "padding": 1,
        "stride": 1
    }, {
        "type": "relu"
    }, {
        "type": "maxpool",
        "kernel_size": 2,
        "stride": 2,
        "padding": 0
    }, {
        "type": "reshape",
        "shape": [-1, 64 * 8 * 8]  # Adjust shape to match the output size
    }, {
        "type": "linear",
        "in_features": 8 * 8 * 64,
        "out_features": 128
    }, {
        "type": "relu"
    }, {
        "type": "linear",
        "in_features": 128,
        "out_features": 10  # CIFAR-10 has 10 classes
    }]
}

model, model_type, optimizer, criterion = ModelFactory().create_model(model_json)

model.load_state_dict(model_params)

torch.save(model, './cache/model/cifar10_model_file.pth')

new_model = torch.load('./cache/model/cifar10_model_file.pth')
new_model.eval()