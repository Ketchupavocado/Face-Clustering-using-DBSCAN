import os
import cv2
import face_recognition

import get_encodings as enc
import face_cluster as cluster


def detect_and_save_faces(image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    face_count = 0
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)

        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_img = rgb_img[top:bottom, left:right]
            save_path = os.path.join(output_dir, f"face_{face_count}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            face_count += 1

    print(f"Total faces detected and saved: {face_count}")
    return output_dir


# === MAIN PIPELINE === #

# Input directory with raw, unprocessed images
raw_images_dir = 'raw_images/'  # <-- you provide this
detected_faces_dir = 'face_images/faces_detected/'

# Step 1: Detect faces and save cropped images
detect_and_save_faces(raw_images_dir, detected_faces_dir)

# Step 2: Create face encodings from the cropped faces
dir_name, data_len = enc.create_face_database(detected_faces_dir)

# Step 3: Perform face clustering
unq_faces, res_dir = cluster.do_cluster(dir_name)

# Output results
print("Number of unique faces detected:", unq_faces)
print(f'Clustered faces saved in: "{res_dir}"')
