import os

# Define the base directory containing the Train folder
base_dir = 'ASL_Dataset/Train/'

# Define the number of images to keep per subfolder
max_images_to_keep = 1000

def keep_limited_images(base_dir, max_images):
    # Iterate over each subfolder in the base directory
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # List all image files (common image extensions)
            images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            # Sort images to keep consistent order
            images.sort()
            # If more than max_images, delete the extras
            if len(images) > max_images:
                images_to_delete = images[max_images:]
                for img in images_to_delete:
                    os.remove(os.path.join(subfolder_path, img))
                print(f"Kept {max_images} images and deleted {len(images_to_delete)} images in {subfolder_path}")
            else:
                print(f"{subfolder_path} has {len(images)} images, no deletion needed")

# Call the function
keep_limited_images(base_dir, max_images_to_keep)
