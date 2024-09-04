import os
import shutil
import cv2
import numpy as np
import random
from tqdm import tqdm
import logging

# Set up logging for detailed debugging and information
logging.basicConfig(
    filename='augmentation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def count_labels(file_path):
    """Calculate the number of class 0 and class 1 labels in each txt file."""
    count_0, count_1 = 0, 0
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            count_0 = sum(1 for line in lines if line.startswith('0'))
            count_1 = sum(1 for line in lines if line.startswith('1'))
    except Exception as e:
        logging.error(f"Error counting labels in {file_path}: {e}")
    return count_0, count_1

def sort_files_by_label_count(folder_path):
    """Sort files based on the count of class 1 and class 0 labels."""
    try:
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    except Exception as e:
        logging.error(f"Error reading directory {folder_path}: {e}")
        return []

    label_counts = []

    for file in files:
        file_path = os.path.join(folder_path, file)
        count_0, count_1 = count_labels(file_path)
        label_counts.append((file, count_0, count_1))

    # Sort by count of class 1 descending, then class 0 ascending
    sorted_files = sorted(label_counts, key=lambda x: (-x[2], x[1]))

    return sorted_files

def adjust_ratio(sorted_files, ratio, input_folder, output_folder):
    """Adjust the ratio of class 0 and class 1 labels and save results to a new folder."""
    total_count_1 = sum(count_1 for _, _, count_1 in sorted_files)
    target_count_0 = int(total_count_1 * ratio)

    kept_files = []
    deleted_files = []

    kept_count_0 = 0
    kept_count_1 = 0

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Keep files prioritizing those with more class 1 annotations
    for file, count_0, count_1 in sorted_files:
        if kept_count_1 + count_1 <= total_count_1 and kept_count_0 + count_0 <= target_count_0:
            kept_files.append((file, count_0, count_1))
            kept_count_0 += count_0
            kept_count_1 += count_1
        else:
            deleted_files.append((file, count_0, count_1))

    # Save kept files and log deleted ones
    try:
        with open(os.path.join(output_folder, 'results.txt'), 'w') as result_file:
            result_file.write(f"Total files: {len(sorted_files)}\n")
            result_file.write(f"Kept files: {len(kept_files)}\n")
            result_file.write(f"Deleted files: {len(deleted_files)}\n")
            result_file.write(f"Desired ratio: {ratio}\n")
            result_file.write(f"Kept class 1 count: {kept_count_1}\n")
            result_file.write(f"Kept class 0 count: {kept_count_0}\n")
            result_file.write("\n")

            for file, count_0, count_1 in kept_files:
                shutil.copy(os.path.join(input_folder, file), os.path.join(output_folder, file))
                result_file.write(f"Kept: {file} | Class 0: {count_0}, Class 1: {count_1}\n")

            result_file.write("\n")
            for file, count_0, count_1 in deleted_files:
                result_file.write(f"Deleted: {file} | Class 0: {count_0}, Class 1: {count_1}\n")
    except Exception as e:
        logging.error(f"Error writing results to {output_folder}: {e}")

    print(f"Total files: {len(sorted_files)}")
    print(f"Kept files: {len(kept_files)}")
    print(f"Deleted files: {len(deleted_files)}")
    print(f"Desired ratio: {ratio}")
    print(f"Kept class 1 count: {kept_count_1}")
    print(f"Kept class 0 count: {kept_count_0}")

    return kept_files, deleted_files

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

def augment_images_and_labels(image_folder, label_folder, output_folder, target_class=1, original_class_1_count=0):
    """Augment images and labels by rotating and saving them to the specified output folder structure."""
    print("Starting augmentation process...")
    logging.info("Augmentation process started.")

    # Create the output directories within the deleted folder
    augmentation_output_folder = os.path.join(output_folder, 'augmented_output')
    for subfolder in ['images', 'labels', 'all', 'image_with_bounding_box']:
        os.makedirs(os.path.join(augmentation_output_folder, subfolder), exist_ok=True)

    # Get list of image files for progress bar
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    current_class_1_count = 0
    augmented_file_count = 0

    # Shuffle image files to randomize the augmentation process
    random.shuffle(image_files)

    # Iterate over all images in the image folder with a progress bar
    for image_filename in tqdm(image_files, desc="Augmenting images", unit="image"):
        if current_class_1_count >= original_class_1_count:
            break

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

        # Check if the image contains the target class for augmentation
        contains_target_class = False
        with open(label_path, 'r') as file:
            for line in file:
                if line.startswith(str(target_class)):
                    contains_target_class = True
                    break

        if not contains_target_class:
            continue  # Skip augmentation if target class is not present

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

            current_class_1_count += sum(1 for label in new_labels if label.startswith(str(target_class)))
            augmented_file_count += 1

        except Exception as e:
            logging.error(f"Error processing label file {label_path}: {e}")
            continue

        # Define output paths
        base_filename = image_filename.replace('.jpg', f'_rotated_{angle:.2f}')
        output_image_path = os.path.join(augmentation_output_folder, 'images', base_filename + '.jpg')
        output_label_path = os.path.join(augmentation_output_folder, 'labels', base_filename + '.txt')
        output_all_image_path = os.path.join(augmentation_output_folder, 'all', base_filename + '.jpg')
        output_all_label_path = os.path.join(augmentation_output_folder, 'all', base_filename + '.txt')
        output_image_with_boxes_path = os.path.join(augmentation_output_folder, 'image_with_bounding_box', base_filename + '_with_boxes.jpg')

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
    print(f"Class 1 count after augmentation: {current_class_1_count}")
    print(f"Number of augmented files: {augmented_file_count}")
    logging.info(f"Class 1 count after augmentation: {current_class_1_count}")
    logging.info(f"Number of augmented files: {augmented_file_count}")

def delete_unmatched_images(txt_folder, jpg_folder):
    """
    Delete images in the jpg_folder that do not have a corresponding txt file in the txt_folder.
    
    Parameters:
    txt_folder (str): The folder containing txt files.
    jpg_folder (str): The folder containing jpg files.
    """
    try:
        # Get set of txt file names without extensions
        txt_filenames = os.listdir(txt_folder)
        txt_numbers = {os.path.splitext(filename)[0] for filename in txt_filenames}

        # Get jpg file names and delete those without matching txt files
        jpg_filenames = os.listdir(jpg_folder)

        for jpg_filename in jpg_filenames:
            jpg_number = os.path.splitext(jpg_filename)[0]
            jpg_path = os.path.join(jpg_folder, jpg_filename)

            if jpg_number not in txt_numbers:
                try:
                    os.remove(jpg_path)
                    logging.info(f"Deleted: {jpg_filename}")
                except Exception as e:
                    logging.error(f"Error deleting file {jpg_filename}: {e}")

    except Exception as e:
        logging.error(f"Error deleting unmatched images: {e}")

    logging.info("Deletion process completed.")

# Main logic execution
def main():
    folder_path = r'C:\personal file\work\TKU\news\train_val label'
    image_folder = r'C:\personal file\work\TKU\news\train_val image'
    output_folder = r'deleted-image-ratio0.1'
    ratio = 0.1  # Adjust this ratio to the desired class 0 to class 1 ratio

    # Step 1: Sort files by label counts
    sorted_files = sort_files_by_label_count(folder_path)

    # Step 2: Count the original class 0 and class 1 labels before deletion
    original_class_1_count = sum(count_1 for _, _, count_1 in sorted_files)
    original_class_0_count = sum(count_0 for _, count_0, _ in sorted_files)

    # Print original counts before any deletion
    print(f"Total files: {len(sorted_files)}")
    print(f"Original class 1 count: {original_class_1_count}")
    print(f"Original class 0 count: {original_class_0_count}")

    # Step 3: Adjust the ratio and keep only necessary files
    kept_files, deleted_files = adjust_ratio(sorted_files, ratio, folder_path, output_folder)

    # Step 4: Perform augmentation to restore the class 1 count back to the original value
    augment_images_and_labels(image_folder, output_folder, output_folder, target_class=1, original_class_1_count=original_class_1_count)

    # Step 5: Delete unmatched images
    delete_unmatched_images(output_folder, image_folder)

if __name__ == "__main__":
    main()
