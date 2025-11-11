import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)

class ContrastiveLearningModel(nn.Module):
    def __init__(self, image_feature_extractor, image_feature_dim, clinical_feature_dim,
                 projection_dim=128, temperature=0.5):
        super(ContrastiveLearningModel, self).__init__()
        self.image_feature_extractor = image_feature_extractor
        self.image_projector = ProjectionHead(
            input_dim=image_feature_dim,
            hidden_dim=512,
            output_dim=projection_dim
        )
        self.clinical_projector = ProjectionHead(
            input_dim=clinical_feature_dim,
            hidden_dim=256,
            output_dim=projection_dim
        )
        self.temperature = temperature

    def forward_image(self, images):
        with torch.no_grad():
            image_features = self.image_feature_extractor(images)
        image_projections = self.image_projector(image_features)
        return image_features, image_projections

    def forward_clinical(self, clinical_features):
        clinical_projections = self.clinical_projector(clinical_features)
        return clinical_features, clinical_projections

    def forward(self, images, clinical_features):
        _, image_projections = self.forward_image(images)
        _, clinical_projections = self.forward_clinical(clinical_features)
        image_projections_norm = nn.functional.normalize(image_projections, dim=1)
        clinical_projections_norm = nn.functional.normalize(clinical_projections, dim=1)
        return image_projections_norm, clinical_projections_norm

def load_projection_model(model_path, image_feature_extractor, image_feature_dim, clinical_feature_dim):
    model = ContrastiveLearningModel(
        image_feature_extractor=image_feature_extractor,
        image_feature_dim=image_feature_dim,
        clinical_feature_dim=clinical_feature_dim,
        projection_dim=128,
        temperature=0.1
    )
    print(f"Load the projection head model: {model_path}")
    try:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=True)
        try:
            image_projector_params = {k.replace('image_projector.', ''): v for k, v in state_dict.items()
                                      if 'image_projector' in k}
            clinical_projector_params = {k.replace('clinical_projector.', ''): v for k, v in state_dict.items()
                                         if 'clinical_projector' in k}
            if image_projector_params and clinical_projector_params:
                model.image_projector.load_state_dict(image_projector_params)
                model.clinical_projector.load_state_dict(clinical_projector_params)
                print("Successfully loaded image and clinical projection head parameters individually")
        except Exception as e:
            print(f"Failed to load projection head parameters individually, but the full model was loaded: {e}")
    except Exception as e:
        print(f"Failed to load projection head model: {e}")
        print("Proceeding with untrained projection heads")
    for param in model.parameters():
        param.requires_grad = False
    return model
