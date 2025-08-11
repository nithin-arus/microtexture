import numpy as np
import cv2
from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import disk
from preprocess.preprocess import preprocess_image

def fractal_dimension(Z):
    """
    Calculate the fractal dimension using box-counting method.
    Input must be a binary 2D NumPy array.
    """
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k * k))[0])

    Z = (Z < 255)  # invert image: 1s for structure, 0s for background
    p = min(Z.shape)
    n = 2 ** np.floor(np.log2(p)).astype(int)
    sizes = 2 ** np.arange(1, int(np.log2(n)))

    counts = [boxcount(Z, size) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

    return -coeffs[0]

def extract_features(image_path):
    gray, binary = preprocess_image(image_path)

    # Fractal Dimension
    fd = fractal_dimension(binary)

    # Entropy
    ent = sk_entropy(gray, disk(5)).mean()

    # Edge Density (Canny)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # Mean + Std Deviation
    mean_val, std_val = cv2.meanStdDev(gray)
    mean_intensity = mean_val[0][0]
    std_dev = std_val[0][0]

    return {
        "fractal_dim": round(fd, 4),
        "entropy": round(ent, 4),
        "edge_density": round(edge_density, 4),
        "mean_intensity": round(mean_intensity, 2),
        "std_dev": round(std_dev, 2)
    }

# Test with an image path
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 extract_features.py path/to/image.jpg")
        sys.exit()

    features = extract_features(sys.argv[1])
    print("Extracted Features:")
    for k, v in features.items():
        print(f"{k}: {v}")
