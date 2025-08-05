import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, feature_extractor, projection_head, embedding_dim=128):
        super(ImageClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.projection_head = projection_head

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = torch.flatten(features, 1)
            projections = self.projection_head(features)

        logits = self.classifier(projections)
        return logits, projections

class ClinicalClassifier(nn.Module):
    def __init__(self, projection_head, embedding_dim=128):
        super(ClinicalClassifier, self).__init__()
        self.projection_head = projection_head

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        with torch.no_grad():
            projections = self.projection_head(x)

        logits = self.classifier(projections)
        return logits, projections
