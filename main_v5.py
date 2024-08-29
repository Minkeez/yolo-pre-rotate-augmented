import os
import cv2
import numpy as np
import random
from tqdm import tqdm  # Import tqdm for the progress bar
import logging

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

def augment_images_and_labels(image_folder, label_folder, output_folder):
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

# Example usage with your folders
image_folder = r'C:\personal file\work\TKU\news\train-image'
label_folder = r'C:\personal file\work\TKU\news\train-label'
output_folder = 'test_output_3'
augment_images_and_labels(image_folder, label_folder, output_folder)
