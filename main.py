import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from utils import get_images_from_folder_S1, get_images_from_folder_S2, get_labels_from_folder, get_city_names_from_folder, visualize_data_sizes, basic_crop_to_match, resize_to_min, center_crop_to_match

########################################################
# DATA LOADING
########################################################

_THIS_FILE = Path(__file__)
ROOT_DIR = _THIS_FILE.parent
DATA_DIR = ROOT_DIR / "data"
S1_PATH = DATA_DIR / "S1"
S2_PATH = DATA_DIR / "S2"
LABELS_PATH = DATA_DIR / "ground_truth"

# on stocke les images du sentinel-1 dans un array de tuple : (image_temps1, image_temps2)
images_s1 = get_images_from_folder_S1(S1_PATH)
# on stocke les images du sentinel-2 dans un array de tuple : (image_temps1, image_temps2)
images_s2 = get_images_from_folder_S2(S2_PATH)
labels = get_labels_from_folder(LABELS_PATH)
city_names = get_city_names_from_folder(LABELS_PATH)

#reshape les images car certaines paires n'ont pas exactement la même taille
images_s1 = [center_crop_to_match(img1, img2) for img1, img2 in images_s1]
images_s2 = [center_crop_to_match(img1, img2) for img1, img2 in images_s2]

########################################################
# CHANGE MAP COMPUTATION
########################################################
def compute_change_map(image_t1, image_t2):
    diff = np.abs(image_t2.astype(np.float32) - image_t1.astype(np.float32))
    #Moyenne de la différence sur toutes les bandes. TODO : étudier les changements sur des bandes spécifiques (des bandes non sensibles aux changement d'illumination ou de végétation)
    diff_gray = np.mean(diff, axis=2)
    #Normalisation entre 0 et 255 pour visualiser
    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Otsu cherche automatiquement le meilleur seuil pour séparer "inchangé" / "changé" dans la carte de changement
    _, change_map = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return change_map

def visualize_change_map(image_t1, image_t2, change_map, ground_truth, satellite, city_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 4, 1)
    plt.title('Image avant (t1)')
    plt.imshow(np.mean(image_t1, axis=2) if satellite == "Sentinel-1" else image_t1, cmap='gray' if satellite == "Sentinel-1" else 'jet')

    plt.subplot(1, 4, 2)
    plt.title('Image après (t2)')
    plt.imshow(np.mean(image_t2, axis=2) if satellite == "Sentinel-1" else image_t2, cmap='gray' if satellite == "Sentinel-1" else 'jet')

    plt.subplot(1, 4, 3)    
    plt.title('Carte de changement (Otsu)')
    plt.imshow(change_map, cmap='gray')

    plt.subplot(1, 4, 4)
    plt.title('Ground truth')
    plt.imshow(ground_truth, cmap='gray')

    plt.suptitle(f"Change detection - {city_name} ({satellite})", fontsize=16)
    plt.show()


for (image_s1_t1, image_s1_t2),(image_s2_t1, image_s2_t2), ground_truth, city_name in zip(images_s1, images_s2, labels, city_names):
    change_map_s1 = compute_change_map(image_s1_t1, image_s1_t2)
    change_map_s2 = compute_change_map(image_s2_t1, image_s2_t2)
    visualize_change_map(image_s1_t1, image_s1_t2, change_map_s1, ground_truth, "Sentinel-1", city_name)
    visualize_change_map(image_s2_t1, image_s2_t2, change_map_s2, ground_truth, "Sentinel-2", city_name)
