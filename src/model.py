import torch.nn as nn
import segmentation_models_pytorch as smp


class SiameseUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet"):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=4,   # 2 channels pre + 2 channels post
            classes=1,
        )

    def forward(self, x):
        return self.model(x)