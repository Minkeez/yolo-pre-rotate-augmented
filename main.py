import cv2
import numpy as np
import random

def rotate_image(image, angle):
    # Get the image size
    (h, w) = image.shape[:2]
    # Calculate the center of the image
    center = (w / 2, h / 2)
    
    # Rotate the image by the given angle
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated_image, M

def rotate_bounding_box(x, y, w, h, M):
    # Get the original corner points of the bounding box
    corners = np.array([
        [x, y],
        [x + w, y],  # Fixed: Added missing comma here
        [x, y + h],
        [x + w, y + h]
    ])
    
    # Convert to homogeneous coordinates (for affine transformation)
    corners = np.hstack([corners, np.ones((4, 1))])
    
    # Rotate the corners
    rotated_corners = M.dot(corners.T).T
    
    # Find the new bounding box
    x_min, y_min = rotated_corners[:, 0].min(), rotated_corners[:, 1].min()
    x_max, y_max = rotated_corners[:, 0].max(), rotated_corners[:, 1].max()
    
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

def is_within_frame(x, y, w, h, frame_width, frame_height):
    # Check if the bounding box is within the image frame
    if x < 0 or y < 0 or (x + w) > frame_width or (y + h) > frame_height:
        return False
    return True

def augment_image_and_label(image_path, label_path):
    # Load image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Generate a random rotation angle between -45 and 45 degrees
    angle = random.uniform(-45, 45)
    
    # Rotate image
    rotated_image, M = rotate_image(image, angle)
    
    # Load label
    new_labels = []
    with open(label_path, 'r') as file:
        for line in file:
            label = line.strip().split()
            class_id = label[0]
            x, y, width, height = map(float, label[1:])
            
            # Convert normalized coordinates to absolute
            x_abs = int(x * w)
            y_abs = int(y * h)
            width_abs = int(width * w)
            height_abs = int(height * h)
            
            # Rotate bounding box
            x_rot, y_rot, w_rot, h_rot = rotate_bounding_box(x_abs, y_abs, width_abs, height_abs, M)
            
            # Check if the rotated bounding box is within the frame
            if is_within_frame(x_rot, y_rot, w_rot, h_rot, w, h):
                # Convert back to normalized coordinates
                x_norm = x_rot / w
                y_norm = y_rot / h
                w_norm = w_rot / w
                h_norm = h_rot / h
                new_labels.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Save the augmented image and labels
    output_image_path = image_path.replace('.jpg', f'_rotated_{angle:.2f}.jpg')
    output_label_path = label_path.replace('.txt', f'_rotated_{angle:.2f}.txt')
    
    cv2.imwrite(output_image_path, rotated_image)
    with open(output_label_path, 'w') as file:
        for label in new_labels:
            file.write(label + '\n')

# Example usage with one image and label
image_path = 'frame_360.jpg'
label_path = 'frame_360.txt'
augment_image_and_label(image_path, label_path)
