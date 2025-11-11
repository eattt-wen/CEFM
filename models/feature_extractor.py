import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

class ImageFeatureExtractor(nn.Module):
    def __init__(self, vit_model: nn.Module):
        super().__init__()
        if hasattr(vit_model, 'base_model'):
            vit_model = vit_model.base_model
        self.vit = vit_model

        if hasattr(self.vit, 'head'):
            self.vit.head = nn.Identity()
        if hasattr(self.vit, 'heads') and hasattr(self.vit.heads, 'head'):
            self.vit.heads.head = nn.Identity()
        if hasattr(self.vit, 'classification_head'):
            self.vit.classification_head = nn.Identity()
        if hasattr(self.vit, 'fc_norm'):
            self.vit.fc_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vit, 'forward_features'):
            feats = self.vit.forward_features(x)
        else:
            feats = self.vit(x)
        if feats.ndim == 3:
            feats = feats[:, 0]
        return feats


def load_image_model(model_path):
    print(f"Load model weights: {model_path}")

    class CustomViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = models.vit_b_16(weights=None)
            in_feats = self.base_model.heads.head.in_features
            self.base_model.heads.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_feats, 2)
            )

        def forward(self, x):
            return self.base_model(x)

    state = torch.load(model_path, map_location='cpu')

    if not isinstance(state, (dict, OrderedDict)):
        raise RuntimeError(f"[ERROR] Loading failed: expected a state_dict, but got {type(state)}")

    if not any(k.startswith("base_model") for k in state.keys()):
        raise RuntimeError(
            "[ERROR] Load failed: The 'base_model' prefix was not detected in the state_dict."
        )

    model = CustomViT()
    model.load_state_dict(state)
    print(f"Complete loading of the weight file: {model_path}")

    for p in model.base_model.parameters():
        p.requires_grad = False
    print("The complete loading of the weight file has completed and the model parameters have been frozen.")

    return model.base_model
