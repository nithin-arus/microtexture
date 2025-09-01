import cv2

def preprocess_image(path, size=(512, 512)):
    """
    Loads an image, converts to grayscale, resizes, and thresholds it.
    Returns: grayscale image and binary image.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image at path '{path}' not found.")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    return resized, binary

# For testing this file directly
if __name__ == "__main__":
    import sys
    from matplotlib import pyplot as plt

    if len(sys.argv) < 2:
        print("Usage: python3 preprocess.py path/to/image.jpg")
        sys.exit()

    image_path = sys.argv[1]
    gray, binary = preprocess_image(image_path)

    # Show images (optional)
    plt.subplot(1, 2, 1)
    plt.title("Grayscale")
    plt.imshow(gray, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title("Binary Threshold")
    plt.imshow(binary, cmap='gray')
    
    plt.show()
