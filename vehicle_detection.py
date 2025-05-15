from ultralytics import YOLO
import os
from PIL import Image
import numpy as np

# Define paths
MODEL_PATH = r"model/best.pt"  # Path to your YOLO model
IMAGES_DIR = r"images"  # Folder containing input images
OUTPUT_DIR = r"output"  # Folder to save processed images

# Load the YOLO model
model = YOLO(MODEL_PATH)


def process_image(image_path, model):
    
    results = model.predict(source=image_path, save=False,conf=0.05)
    vehicle_count = 0
    emergency_vehicle_count = 0
    accident_detected = False
    detected_vehicles = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0] 
            confidence = box.conf[0]  # confidence scores
            class_id = box.cls[0] 
            label = model.names[int(class_id)]  # class label name

            
            if label in [
                "autorickshaw",
                "bicycle",
                "bus",
                "car",
                "motorcycle",
                "truck",
            ]:
                vehicle_count += 1
                detected_vehicles.append(label)

            if label == "emergency":
                emergency_vehicle_count += 1
                detected_vehicles.append(label)

    
            if label == "accident":
                accident_detected = True

    # Save annotated image
    annotated_image_array = results[0].plot()
    annotated_image = Image.fromarray(
        np.uint8(annotated_image_array)
    )  # Convert to PIL.Image
    annotated_image_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    annotated_image.save(annotated_image_path)

    return {
        "total_vehicles": vehicle_count,
        "emergency_vehicles": emergency_vehicle_count,
        "accident_detected": accident_detected,
        "annotated_image_path": annotated_image_path,
    }


def process_images_in_directory(images_dir, model):
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    results_summary = []
    for image_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_name)
        if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"Processing {image_name}...")
            result = process_image(image_path, model)
            result['image_name'] = image_name
            results_summary.append(result)

    return results_summary


if __name__ == "__main__":
    print("Starting detection...")
    results = process_images_in_directory(IMAGES_DIR, model)

    for idx, result in enumerate(results, start=1):
        print(f"Results for Image {idx}:")
        print(f"  Total Vehicles: {result['total_vehicles']}")
        print(f"  Emergency Vehicles: {result['emergency_vehicles']}")
        print(f"  Accident Detected: {result['accident_detected']}")
        print(f"  Annotated Image Saved At: {result['annotated_image_path']}")