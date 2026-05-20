import os
import nibabel as nib
import torch
import torchio as tio


def resize_nifti_images(input_folder, output_folder, target_size=(256, 256, 448)):
    """
    Resize NIfTI medical images to a fixed target size.

    Parameters
    ----------
    input_folder : str
        Folder containing input NIfTI images.
    output_folder : str
        Folder where resized images will be saved.
    target_size : tuple
        Desired output image dimensions.
    """

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith((".nii", ".nii.gz")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            nifti_image = nib.load(input_path)
            data = nifti_image.get_fdata()

            data = torch.tensor(data, dtype=torch.float32)
            data = data.unsqueeze(0)

            resize_transform = tio.transforms.Resize(
                target_shape=target_size,
                image_interpolation="bspline"
            )

            resized_data = resize_transform(data)

            resized_nifti = nib.Nifti1Image(
                resized_data.squeeze(0).numpy(),
                nifti_image.affine,
                nifti_image.header
            )

            nib.save(resized_nifti, output_path)

            print(f"Resized and saved: {filename}")
