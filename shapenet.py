import os
import random

def random_sample_png(base_path):
    # List of categories
    categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # List to store full paths of selected images
    selected_image_paths = []

    # Loop through each category
    for category in categories:
        category_path = os.path.join(base_path, category)
        objects = [o for o in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, o))]
        
        # Randomly sample 5 objects if there are enough objects, otherwise sample all available
        sampled_objects = random.sample(objects, min(5, len(objects)))

        # For each sampled object, select a random PNG image
        for obj in sampled_objects:
            obj_path = os.path.join(category_path, obj, 'rendering')
            png_files = [f for f in os.listdir(obj_path) if f.endswith('.png')]
            selected_png = random.choice(png_files)
            full_path = os.path.join(obj_path, selected_png)
            selected_image_paths.append(full_path)
    
    return selected_image_paths

# Base path to the data_tf directory
base_path = '/om/user/evan_kim/SculptFormer/datasets/data/shapenet/data_tf'

# Run the function
selected_images = random_sample_png(base_path)

# Print selected images
for image_path in selected_images:
    print(image_path)
