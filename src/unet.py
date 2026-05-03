import torch
import torch.nn as nn


class UNetOriginal(nn.Module):
    """
    Original U-Net paper architecture.

    Important:
    - Use unpadded 3x3 convolutions.
    - Use crop + concat in skip connections.
    - Input 572x572 -> output 388x388.
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # TODO:
        # Encoder:
        # 3 -> 64 -> 64
        # 64 -> 128 -> 128
        # 128 -> 256 -> 256
        # 256 -> 512 -> 512
        #
        # Bottleneck:
        # 512 -> 1024 -> 1024
        #
        # Decoder:
        # upconv 1024 -> 512
        # concat with cropped encoder feature
        # 1024 -> 512 -> 512
        #
        # upconv 512 -> 256
        # concat
        # 512 -> 256 -> 256
        #
        # upconv 256 -> 128
        # concat
        # 256 -> 128 -> 128
        #
        # upconv 128 -> 64
        # concat
        # 128 -> 64 -> 64
        #
        # final 1x1 conv:
        # 64 -> out_channels

        raise NotImplementedError("Implement the original U-Net here.")

    def forward(self, x):
        # TODO:
        # 1. Encoder path
        # 2. Bottleneck
        # 3. Decoder path
        # 4. Crop encoder features before concatenation
        # 5. Return logits, not sigmoid probabilities

        raise NotImplementedError("Implement forward pass.")

    def crop_to_match(self, encoder_feature, decoder_feature):
        """
        Center-crop encoder_feature so it has same H,W as decoder_feature.
        """
        _, _, h, w = decoder_feature.shape
        _, _, H, W = encoder_feature.shape

        top = (H - h) // 2
        left = (W - w) // 2

        return encoder_feature[:, :, top:top + h, left:left + w]