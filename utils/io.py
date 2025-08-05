import os
import torch

def mkdir_if_not_exists(path):
    os.makedirs(path, exist_ok=True)

def save_checkpoint(state, path):
    torch.save(state, path)

def load_state_dict_strict(model, path):
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    return model
