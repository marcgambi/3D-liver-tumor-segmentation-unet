import torch
import matplotlib.pyplot as plt


def visualize_segmentation(output, label, image, slice_start=200, slice_end=250):
    """
    Visualize model output, ground truth label, and input image slice by slice.
    """

    output = output[0]
    label = label[0]
    image = image[0][0]

    num_slices = slice_end - slice_start

    plt.figure(figsize=(15, 5 * num_slices))

    for i in range(num_slices):
        current_slice = slice_start + i

        plt.subplot(num_slices, 3, 3 * i + 1)
        with torch.no_grad():
            plt.imshow(torch.argmax(output.cpu(), dim=0)[:, :, current_slice])
        plt.title(f"Model Output - Slice {current_slice}")
        plt.axis("off")

        plt.subplot(num_slices, 3, 3 * i + 2)
        plt.imshow(label.cpu()[:, :, current_slice])
        plt.title(f"Ground Truth - Slice {current_slice}")
        plt.axis("off")

        plt.subplot(num_slices, 3, 3 * i + 3)
        plt.imshow(image.cpu()[:, :, current_slice], cmap="gray")
        plt.title(f"Input Image - Slice {current_slice}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
