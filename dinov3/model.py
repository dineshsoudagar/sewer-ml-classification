import timm
import torch
import torch.nn as nn

class DinoV3MultiLabel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        # Create as a feature extractor (no classifier)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # removes default head
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


if __name__ == "__main__":
    model = DinoV3MultiLabel(model_name="vit_large_patch16_dinov3.lvd1689m", num_classes=19)
    print(model)
