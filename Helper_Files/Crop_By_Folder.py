from PIL import Image
import os
#Beware: This will replace images in-place with the cropped version.
#Make sure you use an duplicate folder for this to avoid losing data
folder = r"c:\Users\dbrimmer\Downloads\Plant11-Cucumber"
def mass_cropper(folder):
    # Both Variables are by pixel resolution
    crop_width = 640  # set your desired width
    crop_height = 380  # set your desired height

    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            cropped_path = os.path.join(folder, f"cropped_{filename}")
            with Image.open(img_path) as img:
                width, height = img.size
                left = (width - crop_width) // 2
                top = (height - crop_height) // 2
                right = left + crop_width
                bottom = top + crop_height
                # Crop the top portion of the image
                cropped_img = img.crop((0, height - crop_height, width, height))
                cropped_img.save(cropped_path)
            os.remove(img_path)  # Remove the original image

mass_cropper(folder)
