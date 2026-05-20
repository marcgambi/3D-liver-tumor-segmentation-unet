import torch


def dice_loss(pred, target, smooth=1):
    """
    Compute Dice Loss for segmentation tasks.
    """

    pred = torch.softmax(pred, dim=1)

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()
