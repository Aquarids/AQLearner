import torch

type_regression = "regression"
type_binary_classification = "binary_classification"
type_multi_classification = "multi_classification"


class ModelFactory:

    def __init__(self):
        self.layer_creators = {
            "conv1d": self.create_conv1d_layer,
            "conv2d": self.create_conv2d_layer,
            "linear": self.create_linear_layer,
            "maxpool": self.create_maxpool_layer,
            "flatten": self.create_flatten_layer,
            "reshape": self.create_reshape_layer,
            "dropout": self.create_dropout_layer,
            "softmax": self.create_softmax_layer,
            "sigmoid": self.create_sigmoid_layer,
            "relu": self.create_relu_layer,
            "tanh": self.create_tanh_layer,
            "lazy_conv1d": self.create_lazy_conv1d_layer,
            "lazy_conv2d": self.create_lazy_conv2d_layer,
            "lazy_linear": self.create_lazy_linear_layer,
        }

    def create_conv1d_layer(self, layer_info):
        return torch.nn.Conv1d(in_channels=layer_info["in_channels"],
                               out_channels=layer_info["out_channels"],
                               kernel_size=layer_info["kernel_size"],
                               stride=layer_info["stride"],
                               padding=layer_info["padding"])

    def create_conv2d_layer(self, layer_info):
        return torch.nn.Conv2d(in_channels=layer_info["in_channels"],
                               out_channels=layer_info["out_channels"],
                               kernel_size=layer_info["kernel_size"],
                               stride=layer_info["stride"],
                               padding=layer_info["padding"])

    def create_linear_layer(self, layer_info):
        return torch.nn.Linear(in_features=layer_info["in_features"],
                               out_features=layer_info["out_features"])

    def create_maxpool_layer(self, layer_info):
        return torch.nn.MaxPool2d(kernel_size=layer_info["kernel_size"],
                                  stride=layer_info["stride"],
                                  padding=layer_info["padding"])

    def create_flatten_layer(self, layer_info):
        return FlattenLayer()

    def create_reshape_layer(self, layer_info):
        return ReshapeLayer(shape=layer_info["shape"])

    def create_dropout_layer(self, layer_info):
        return torch.nn.Dropout(layer_info["dropout_rate"])

    def create_softmax_layer(self, layer_info):
        return torch.nn.Softmax(dim=layer_info["dim"])

    def create_sigmoid_layer(self, layer_info):
        return torch.nn.Sigmoid()

    def create_relu_layer(self, layer_info):
        return torch.nn.ReLU(inplace=True)

    def create_tanh_layer(self, layer_info):
        return torch.nn.Tanh()

    def create_lazy_conv1d_layer(self, layer_info):
        return torch.nn.Conv1d(in_channels=layer_info["in_channels"],
                               out_channels=layer_info["out_channels"],
                               kernel_size=layer_info["kernel_size"],
                               stride=layer_info["stride"],
                               padding=layer_info["padding"])

    def create_lazy_conv2d_layer(self, layer_info):
        return torch.nn.Conv2d(in_channels=layer_info["in_channels"],
                               out_channels=layer_info["out_channels"],
                               kernel_size=layer_info["kernel_size"],
                               stride=layer_info["stride"],
                               padding=layer_info["padding"])

    def create_lazy_linear_layer(self, layer_info):
        return torch.nn.Linear(in_features=layer_info["in_features"],
                               out_features=layer_info["out_features"])

    def create_optimizer(self, torch_model, optimizer_info, learning_rate):
        if optimizer_info == "adam":
            return torch.optim.Adam(torch_model.parameters(), lr=learning_rate)
        elif optimizer_info == "sgd":
            return torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
        # Add more optimizers here
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_info}")

    def create_criterion(self, criterion):
        if criterion == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        elif criterion == "mse":
            return torch.nn.MSELoss()
        elif criterion == "bce":
            return torch.nn.BCELoss()
        # Add more loss function here
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")

    # {
    # 	"model_type": "regression",


# 	"learning_rate": 0.001,
# 	"optimizer": "sgd",
# 	"criterion": "mse",
# 	"layers": [{
# 		...
# 	}]
# }

    def create_model(self, model_params):
        layers = []
        model_type = model_params["model_type"]
        for layer_info in model_params["layers"]:
            layer_type = layer_info["type"]
            if layer_type in self.layer_creators:
                layer = self.layer_creators[layer_type](layer_info)
                layers.append(layer)
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        model = torch.nn.Sequential(*layers)

        optimizer = self.create_optimizer(model, model_params["optimizer"],
                                          model_params["learning_rate"])
        criterion = self.create_criterion(model_params["criterion"])

        return model, model_type, optimizer, criterion


class FlattenLayer(torch.nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ReshapeLayer(torch.nn.Module):

    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
