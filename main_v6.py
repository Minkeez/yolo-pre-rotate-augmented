import os
import cv2
import numpy as np
import random
from tqdm import tqdm  # Import tqdm for the progress bar
import logging
import shutil

# Set up logging for detailed debugging and information
logging.basicConfig(filename='augmentation.log', level=logging.INFO)

def rotate_image(image, angle):
    """Rotate the image by a given angle and return the rotated image and transformation matrix."""
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated_image, M

def rotate_bounding_box(x_center, y_center, w, h, M):
    """Rotate the bounding box corners based on the transformation matrix and return the new bounding box."""
    corners = np.array([
        [x_center - w / 2, y_center - h / 2],
        [x_center + w / 2, y_center - h / 2],
        [x_center - w / 2, y_center + h / 2],
        [x_center + w / 2, y_center + h / 2]
    ])
    corners = np.hstack([corners, np.ones((4, 1))])  # Convert to homogeneous coordinates
    rotated_corners = M.dot(corners.T).T
    x_min, y_min = rotated_corners[:, 0].min(), rotated_corners[:, 1].min()
    x_max, y_max = rotated_corners[:, 0].max(), rotated_corners[:, 1].max()
    return x_min, y_min, x_max, y_max

def clip_bounding_box(x, y, w, h, frame_width, frame_height):
    """Clip the bounding box to ensure it stays within the frame boundaries."""
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame_width - x)
    h = min(h, frame_height - y)
    return x, y, w, h

def is_within_frame(x, y, w, h, frame_width, frame_height):
    """Check if any part of the bounding box is within the image frame."""
    return not (x + w <= 0 or y + h <= 0 or x >= frame_width or y >= frame_height)

def adaptive_rotation_angle(object_size, distance_from_center):
    """Calculate rotation angle based on object characteristics."""
    if object_size < 0.2:  # Example threshold
        return random.uniform(-30, 30)  # Smaller objects get a larger rotation range
    else:
        return random.uniform(-10, 10)  # Larger objects get a smaller rotation range

def augment_images_and_labels(image_folder, label_folder, output_folder, desired_ratio):
    """Augment images and labels by rotating and saving them to the specified output folder structure."""
    
    print("Starting augmentation process...")
    logging.info("Augmentation process started.")

    # Create the output directories if they don't exist
    for subfolder in ['images', 'labels', 'all', 'image_with_bounding_box']:
        os.makedirs(os.path.join(output_folder, subfolder), exist_ok=True)
    
    # Get list of image files for progress bar
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    
    # Iterate over all images in the image folder with a progress bar
    for image_filename in tqdm(image_files, desc="Augmenting images", unit="image"):
        image_path = os.path.join(image_folder, image_filename)
        label_path = os.path.join(label_folder, image_filename.replace('.jpg', '.txt'))
        
        # Check if label file exists
        if not os.path.exists(label_path):
            logging.warning(f"Label file for {image_filename} not found. Skipping this image.")
            continue
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Could not read image {image_path}. Skipping.")
            continue
        h, w = image.shape[:2]
        
        # Generate a rotation angle based on adaptive function
        angle = adaptive_rotation_angle(1.0, 1.0)  # Use real data here for object size and distance
        
        # Rotate image
        rotated_image, M = rotate_image(image, angle)
        rotated_image_with_boxes = rotated_image.copy()
        
        new_labels = []
        try:
            with open(label_path, 'r') as file:
                for line in file:
                    label = line.strip().split()
                    class_id = label[0]
                    x_center, y_center, width, height = map(float, label[1:])
                    
                    # Convert normalized coordinates to absolute
                    x_center_abs = x_center * w
                    y_center_abs = y_center * h
                    width_abs = width * w
                    height_abs = height * h
                    
                    # Rotate bounding box
                    x_min, y_min, x_max, y_max = rotate_bounding_box(x_center_abs, y_center_abs, width_abs, height_abs, M)
                    
                    # Ensure bounding box stays within image boundaries
                    x_min, y_min, width_abs, height_abs = clip_bounding_box(x_min, y_min, x_max - x_min, y_max - y_min, w, h)
                    
                    # Check visibility of the rotated and clipped bounding box
                    if is_within_frame(x_min, y_min, width_abs, height_abs, w, h):
                        # Convert back to normalized coordinates
                        x_norm = (x_min + width_abs / 2) / w
                        y_norm = (y_min + height_abs / 2) / h
                        w_norm = width_abs / w
                        h_norm = height_abs / h
                        new_labels.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
                        
                        # Draw the clipped bounding box on the rotated image
                        cv2.rectangle(rotated_image_with_boxes, (int(x_min), int(y_min)), (int(x_min + width_abs), int(y_min + height_abs)), (0, 0, 255), 2)
        except Exception as e:
            logging.error(f"Error reading label file {label_path}: {e}")
            continue
        
        # Define output paths
        base_filename = image_filename.replace('.jpg', f'_rotated_{angle:.2f}')
        output_image_path = os.path.join(output_folder, 'images', base_filename + '.jpg')
        output_label_path = os.path.join(output_folder, 'labels', base_filename + '.txt')
        output_all_image_path = os.path.join(output_folder, 'all', base_filename + '.jpg')
        output_all_label_path = os.path.join(output_folder, 'all', base_filename + '.txt')
        output_image_with_boxes_path = os.path.join(output_folder, 'image_with_bounding_box', base_filename + '_with_boxes.jpg')
        
        # Save the augmented images and labels
        cv2.imwrite(output_image_path, rotated_image)
        cv2.imwrite(output_image_with_boxes_path, rotated_image_with_boxes)
        cv2.imwrite(output_all_image_path, rotated_image)
        try:
            with open(output_label_path, 'w') as file:
                for label in new_labels:
                    file.write(label + '\n')
            with open(output_all_label_path, 'w') as file:
                for label in new_labels:
                    file.write(label + '\n')
        except Exception as e:
            logging.error(f"Error writing to label files: {e}")
            continue
    
    print("Augmentation process finished.")
    logging.info("Augmentation process finished.")

    # Adjust label ratio after augmentation
    sorted_files = sort_files_by_label_count(os.path.join(output_folder, 'labels'))
    adjust_ratio(sorted_files, desired_ratio, os.path.join(output_folder, 'adjusted_labels'), os.path.join(output_folder, 'labels'), os.path.join(output_folder, 'images'))

    # Split dataset into training and validation sets
    split_dataset(os.path.join(output_folder, 'all'), os.path.join(output_folder, 'train'), os.path.join(output_folder, 'val'), 0.2)

def count_labels(file_path):
    """Count the number of class 0 and class 1 in each txt file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        count_0 = sum(1 for line in lines if line.startswith('0'))
        count_1 = sum(1 for line in lines if line.startswith('1'))
    return count_0, count_1

def sort_files_by_label_count(folder_path):
    """Sort files by the number of class 1, then by class 0."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    label_counts = []
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        count_0, count_1 = count_labels(file_path)
        label_counts.append((file, count_0, count_1))
    
    # Sort by the number of class 1 descending, then by class 0 ascending
    sorted_files = sorted(label_counts, key=lambda x: (-x[2], x[1]))
    
    return sorted_files

def adjust_ratio(sorted_files, ratio, output_folder, label_folder, image_folder):
    """Adjust the ratio of class 0 and class 1 and save results to a new folder."""
    total_count_1 = sum(count_1 for _, _, count_1 in sorted_files)
    target_count_0 = int(total_count_1 * ratio)
    
    kept_files = []
    deleted_files = []
    
    kept_count_0 = 0
    kept_count_1 = 0
    
    # Prioritize keeping files with more class 1 objects
    for file, count_0, count_1 in sorted_files:
        if kept_count_1 + count_1 <= total_count_1 and kept_count_0 + count_0 <= target_count_0:
            kept_files.append((file, count_0, count_1))
            kept_count_0 += count_0
            kept_count_1 += count_1
        else:
            deleted_files.append((file, count_0, count_1))
    
    # Ensure the new folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save the kept files and delete excess files
    for file, count_0, count_1 in kept_files:
        shutil.copy(os.path.join(label_folder, file), os.path.join(output_folder, file))
    
    # Delete images with no corresponding labels
    for file, count_0, count_1 in deleted_files:
        img_name = file.replace('.txt', '.jpg')
        img_path = os.path.join(image_folder, img_name)
        if os.path.exists(img_path):
            os.remove(img_path)

    print("Ratio adjustment and deletion completed.")

def split_dataset(source_folder, train_folder, val_folder, val_ratio):
    """Split dataset into training and validation sets based on a specified ratio."""
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Get all jpg and txt files from the source folder
    jpg_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]
    txt_files = [f for f in os.listdir(source_folder) if f.endswith('.txt')]

    # Extract base filenames without extensions
    jpg_files_base = {os.path.splitext(f)[0] for f in jpg_files}
    txt_files_base = {os.path.splitext(f)[0] for f in txt_files}

    # Find common base filenames that have both jpg and txt files
    common_files_base = list(jpg_files_base & txt_files_base)

    # Determine the number of files for the validation set
    val_size = int(len(common_files_base) * val_ratio)

    # Randomly select files for the validation set
    val_files_base = random.sample(common_files_base, val_size)

    # Remaining files will be for the training set
    train_files_base = set(common_files_base) - set(val_files_base)

    # Function to move files to the respective folder
    def move_files(files_base, src_folder, dest_folder):
        for base_name in files_base:
            jpg_file = base_name + '.jpg'
            txt_file = base_name + '.txt'
            
            jpg_source_path = os.path.join(src_folder, jpg_file)
            txt_source_path = os.path.join(src_folder, txt_file)
            
            jpg_dest_path = os.path.join(dest_folder, jpg_file)
            txt_dest_path = os.path.join(dest_folder, txt_file)
            
            if os.path.exists(jpg_source_path) and os.path.exists(txt_source_path):
                shutil.move(jpg_source_path, jpg_dest_path)
                shutil.move(txt_source_path, txt_dest_path)

    # Move validation files
    move_files(val_files_base, source_folder, val_folder)

    # Move training files
    move_files(train_files_base, source_folder, train_folder)

    print("Training and validation split completed!")

# Usage
image_folder = r'C:\personal file\work\TKU\news\train-image'
label_folder = r'C:\personal file\work\TKU\news\train-label'
output_folder = 'test_output_4'
desired_ratio = 1.0  # Adjust ratio here

augment_images_and_labels(image_folder, label_folder, output_folder, desired_ratio)