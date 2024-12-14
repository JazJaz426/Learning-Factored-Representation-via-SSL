from models.ssl_models.create_nn import NatureCNN, resnet9
import torch

model_modules = {
    "NatureCNN": NatureCNN,
    "resnet9": resnet9,
}

def load_factored_model(model_name, model_ckpt):
    Module = model_modules[model_name]
    model = Module()
    model.load_state_dict(torch.load(model_ckpt)['backbone'])
    return model