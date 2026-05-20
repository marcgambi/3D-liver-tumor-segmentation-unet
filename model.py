from monai.networks.nets import UNet


def build_unet_model(device):
    """
    Build a 3D U-Net model for medical image segmentation.
    """

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.3
    ).to(device)

    return model
