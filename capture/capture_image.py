from picamera2 import Picamera2
from datetime import datetime
import os
import time

def capture_image(label="sample", base_dir="capture/images"):
    """Capture an image using the Raspberry Pi camera"""
    folder = os.path.join(base_dir, label)
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.jpg"
    filepath = os.path.join(folder, filename)

    try:
        picam2 = Picamera2()
        picam2.start()
        time.sleep(2)  # Allow camera to warm up
        picam2.capture_file(filepath)
        picam2.close()
        print(f"Picture saved at: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None

def capture_multiple_images(label, count=1, delay=1, base_dir="capture/images"):
    """Capture multiple images with a delay between captures"""
    captured_files = []
    for i in range(count):
        if i > 0:
            time.sleep(delay)
        filepath = capture_image(label, base_dir)
        if filepath:
            captured_files.append(filepath)
            print(f"Captured {i+1}/{count}: {os.path.basename(filepath)}")
    return captured_files

if __name__ == "__main__":
    label = input("Enter material label: ")
    count = int(input("Number of images to capture (default 1): ") or "1")
    capture_multiple_images(label, count)