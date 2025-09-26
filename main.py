import os
import cv2
import numpy as np
from skimage.filters import gabor
from skimage.filters.rank import entropy
from skimage.morphology import disk, dilation, erosion
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

input_folder = "data" 
output_folder = "output" 
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4] 
frequencies = [0.1, 0.2, 0.3]  

os.makedirs(output_folder, exist_ok=True)

# Helper function to compute Gabor feature maps
def compute_gabor_features(image):
    gabor_features = []
    for frequency in frequencies:
        for theta in orientations:
            real, _ = gabor(image, frequency=frequency, theta=theta)
            gabor_features.append(real)
    return np.stack(gabor_features, axis=-1)

# Helper function to calculate local entropy
def calculate_local_entropy(image):
    return entropy(image.astype(np.uint8), disk(55))

# Helper function to apply morphological operations
def apply_morphology(segmented_image):
    structuring_element = disk(10)
    morphed_image = dilation(segmented_image, structuring_element)
    morphed_image = erosion(morphed_image, structuring_element)
    return morphed_image

image_files = [f for f in os.listdir(input_folder) if f.startswith('tm') and f.endswith('.png')]

image_files.sort(key=lambda x: int(x[2:-4]))

for idx, image_file in enumerate(image_files, 1):

    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    num_clusters = 6

    gabor_features = compute_gabor_features(image)

    entropy_maps = np.stack([calculate_local_entropy(gf) for gf in gabor_features.transpose(2, 0, 1)], axis=-1)

    feature_vector = entropy_maps.reshape(-1, entropy_maps.shape[-1])

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(feature_vector)
    segmented_image = labels.reshape(image.shape)

    morphed_image = apply_morphology(segmented_image)

    gt_file = f"gt{image_file[2:-6]}.png" 
    gt_path = os.path.join(input_folder, gt_file)
    print(f"gt{image_file[2:-6]}.png")
    if os.path.exists(gt_path):
        ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).flatten()
        ari_score = adjusted_rand_score(ground_truth, labels)
    else:
        ground_truth = None
        ari_score = -1  

    segmented_image_path = os.path.join(output_folder, f"seg{idx}_1_1.png")
    # morphed_image_path = os.path.join(output_folder, f"morphed{idx}_1_1.png")
    # entropy_map_path = os.path.join(output_folder, f"entropy{idx}_1_1.png")

    cv2.imwrite(segmented_image_path, segmented_image.astype(np.uint8) * (255 // num_clusters))  # Scale to 0-255
    # cv2.imwrite(morphed_image_path, morphed_image.astype(np.uint8) * (255 // num_clusters))  # Scale to 0-255
    # cv2.imwrite(entropy_map_path, (entropy_maps.mean(axis=-1) * 255).astype(np.uint8))  # Scale to 0-255

    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 4, 1)
    # plt.title('Original Image')
    # plt.imshow(image, cmap='gray')

    # plt.subplot(1, 4, 2)
    # plt.title('Segmented Image')
    # plt.imshow(segmented_image, cmap='jet')

    # plt.subplot(1, 4, 3)
    # plt.title('Entropy Feature Map')
    # plt.imshow(entropy_maps.mean(axis=-1), cmap='viridis')

    # plt.subplot(1, 4, 4)
    # plt.title('Morphed Image')
    # plt.imshow(morphed_image, cmap='jet')

    # plt.suptitle(f'File: {image_file}, ARI: {ari_score:.2f}')

    # plt.show()
