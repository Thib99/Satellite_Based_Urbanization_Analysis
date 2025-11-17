from ast import And
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm

from utils import (
    get_images_from_folder_S1, 
    get_images_from_folder_S2, 
    get_B_from_folder_S2, 
    get_labels_from_folder, 
    get_city_names_from_folder, 
    visualize_data_sizes, 
    basic_crop_to_match, 
    resize_to_min, 
    center_crop_to_match, 
    upsample_images_A, 
    plot_one_image, 
    normalize_image, 
    fitAtoB)

########################################################
# DATA LOADING
########################################################

_THIS_FILE = Path(__file__)
ROOT_DIR = _THIS_FILE.parent
DATA_DIR = ROOT_DIR / "data"
S1_PATH = DATA_DIR / "S1"
S2_PATH = DATA_DIR / "S2"
LABELS_PATH = DATA_DIR / "ground_truth"

images_s1 = get_images_from_folder_S1(S1_PATH) # Chaque image a 2 channels : VV, VH
images_s2_RGB = get_images_from_folder_S2(S2_PATH) # Chaque image a 3 channels : R, G, B
images_s2_B11 = get_B_from_folder_S2(S2_PATH, "11") #Chaque image a 1 channel : B11
images_s2_B8 = get_B_from_folder_S2(S2_PATH, "08") #Chaque image a 1 channel : B8
images_s2_B4 = get_B_from_folder_S2(S2_PATH, "04") #Chaque image a 1 channel : B4
images_s2_B12 = get_B_from_folder_S2(S2_PATH, "12") #Chaque image a 1 channel : B12
labels = get_labels_from_folder(LABELS_PATH)
city_names = get_city_names_from_folder(LABELS_PATH)

########################################################
# DATA PREPROCESSING
########################################################

#normalisation des images
#images_s1 = [normalize_image(img, ?) for img in images_s1 for pair_image in img]
images_s2_RGB = [(normalize_image(img1, 255), normalize_image(img2, 255)) for img1, img2 in images_s2_RGB]
images_s2_B11 = [(normalize_image(img1, 10000), normalize_image(img2, 10000)) for img1, img2 in images_s2_B11]
images_s2_B8 = [(normalize_image(img1, 10000), normalize_image(img2, 10000)) for img1, img2 in images_s2_B8]
images_s2_B4 = [(normalize_image(img1, 10000), normalize_image(img2, 10000)) for img1, img2 in images_s2_B4]
images_s2_B12 = [(normalize_image(img1, 10000), normalize_image(img2, 10000)) for img1, img2 in images_s2_B12]

#reshape les images car certaines paires n'ont pas exactement la même taille
images_s1 = [center_crop_to_match(img1, img2) for img1, img2 in images_s1]
images_s2_RGB = [center_crop_to_match(img1, img2) for img1, img2 in images_s2_RGB]
images_s2_B11 = [center_crop_to_match(img1, img2) for img1, img2 in images_s2_B11]
images_s2_B8 = [center_crop_to_match(img1, img2) for img1, img2 in images_s2_B8]
images_s2_B4 = [center_crop_to_match(img1, img2) for img1, img2 in images_s2_B4]
images_s2_B12 = [center_crop_to_match(img1, img2) for img1, img2 in images_s2_B12]
images_s2_B11 = upsample_images_A(images_s2_B11, images_s2_B8) # Upsample les images de B11 pour qu'elles aient la même taille que les images de B8
images_s2_B4 = upsample_images_A(images_s2_B4, images_s2_B8) # Upsample les images de B4 pour qu'elles aient la même taille que les images de B8
images_s2_B12 = upsample_images_A(images_s2_B12, images_s2_B8) # Upsample les images de B12 pour qu'elles aient la même taille que les images de B8
########################################################
# FEATURE ENGINEERING
########################################################

# NBDI 
images_s2_build_up_index = []  #on étudie les changements sur des bandes spécifiques (bandes non sensibles aux changement d'illumination ou de végétation)
for (pair11, pair8) in zip(images_s2_B11, images_s2_B8):
    i11_t1, i11_t2 = pair11
    i8_t1,  i8_t2  = pair8

    idx_t1 = (i11_t1 - i8_t1) / (i11_t1 + i8_t1 + 1e-6)
    idx_t2 = (i11_t2 - i8_t2) / (i11_t2 + i8_t2 + 1e-6)

    images_s2_build_up_index.append((idx_t1, idx_t2))

images_s2_build_up_index_t2 = [pair[1] for pair in images_s2_build_up_index]
masks_build_up_index = [(img > 0) for img in images_s2_build_up_index_t2]
masks_build_up_index = [np.squeeze(img, axis=2) for img in masks_build_up_index]
masks_build_up_index = [mask.astype(int) for mask in masks_build_up_index]

# CVA
images_s2_B4 = [(np.squeeze(img1, axis=2), np.squeeze(img2, axis=2)) for img1, img2 in images_s2_B4]

CVAs = []
for (pair4, pair11, pair12) in zip(images_s2_B4, images_s2_B11, images_s2_B12):
    i4_t1, i4_t2 = pair4
    i11_t1, i11_t2 = pair11
    i12_t1, i12_t2 = pair12

    V1 = np.stack([i4_t1, i11_t1, i12_t1], axis=-1)
    V2 = np.stack([i4_t2, i11_t2, i12_t2], axis=-1)
    V1, V2 = np.squeeze(V1), np.squeeze(V2)
    CVAs.append((V1, V2))

########################################################
# CHANGE MAP COMPUTATION
########################################################
def compute_change_map(image_t1: np.ndarray, image_t2: np.ndarray, int_distance : int = 2)->tuple[np.ndarray, np.ndarray]:
    if int_distance == 1 : 
        diff = np.abs(image_t2.astype(np.float32) - image_t1.astype(np.float32))
        # Moyenne de la différence sur toutes les bandes si l'image est multi-bandes
        diff_gray = np.mean(diff, axis=-1)
    elif int_distance == 2 :
        # Norme vectorielle sur l'axe des canaux (déjà une image 2D)
        diff_gray = np.linalg.norm(image_t2.astype(np.float32) - image_t1.astype(np.float32), axis=-1)
    #Normalisation entre 0 et 255 pour visualiser
    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    diff_smooth = cv2.GaussianBlur(diff_norm, (5,5), 1)

    # Otsu cherche automatiquement le meilleur seuil pour séparer "inchangé" / "changé" dans la carte de changement
    _, change_map = cv2.threshold(diff_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return diff_norm,change_map

def visualize_change_map(image_t1: np.ndarray, image_t2: np.ndarray, diff_norm: np.ndarray, change_map: np.ndarray, ground_truth: np.ndarray, mode: str, city_name: str, title: str):
    plt.figure(figsize=(12, 5))
    if mode == "RGB":
        plt.subplot(1, 5, 1)
        plt.title('Image avant (t1)')
        plt.imshow(image_t1, cmap='jet')
        plt.subplot(1, 5, 2)
        plt.title('Image après (t2)')
        plt.imshow(image_t2, cmap='jet')
        plt.subplot(1, 5, 3)
        plt.title('Différence normale')
        plt.imshow(diff_norm, cmap='gray')
        plt.subplot(1, 5, 4)
        plt.title('Carte de changement (Otsu)')
        plt.imshow(change_map, cmap='gray')
        plt.subplot(1, 5, 5)
        plt.title('Ground truth')
        plt.imshow(ground_truth, cmap='gray')
        plt.suptitle(f"Change detection - {city_name} ({title})", fontsize=16)
        plt.show()
    elif mode == "B":
        plt.subplot(1, 5, 1)
        plt.title('Image avant (t1)')
        plt.imshow(image_t1, cmap='gray')
        plt.subplot(1, 5, 2)
        plt.title('Image après (t2)')
        plt.imshow(image_t2, cmap='gray')
        plt.subplot(1, 5, 3)    
        plt.title('Différence normale')
        plt.imshow(diff_norm, cmap='gray')
        plt.subplot(1, 5, 4)
        plt.title('Carte de changement (Otsu)')
        plt.imshow(change_map, cmap='gray')
        plt.subplot(1, 5, 5)
        plt.title('Ground truth')
        plt.imshow(ground_truth, cmap='gray')
        plt.suptitle(f"Change detection - {city_name} ({title})", fontsize=16)
        plt.show()
    elif mode == "S1":
        plt.subplot(1, 5, 1)
        plt.title('Image avant (t1)')
        plt.imshow(np.mean(image_t1, axis=-1), cmap='gray')
        plt.subplot(1, 5, 2)
        plt.title('Image après (t2)')
        plt.imshow(np.mean(image_t2, axis=-1), cmap='gray')
        plt.subplot(1, 5, 3)    
        plt.title('Différence normale')
        plt.imshow(diff_norm, cmap='gray')
        plt.subplot(1, 5, 4)
        plt.title('Carte de changement (Otsu)')
        plt.imshow(change_map, cmap='gray')
        plt.subplot(1, 5, 5)
        plt.title('Ground truth')
        plt.imshow(ground_truth, cmap='gray')
        plt.suptitle(f"Change detection - {city_name} ({title})", fontsize=16)
        plt.show()
    elif mode == "CVA":
        plt.subplot(1, 5, 1)
        plt.title('Image avant (t1)')
        plt.imshow(image_t1, cmap='jet')
        plt.subplot(1, 5, 2)
        plt.title('Image après (t2)')
        plt.imshow(image_t2, cmap='jet')
        plt.subplot(1, 5, 3)
        plt.title('Différence normale')
        plt.imshow(diff_norm, cmap='gray')
        plt.subplot(1, 5, 4)
        plt.title('Carte de changement (Otsu)')
        plt.imshow(change_map, cmap='gray')
        plt.subplot(1, 5, 5)
        plt.title('Ground truth')
        plt.imshow(ground_truth, cmap='gray')
        plt.suptitle(f"Change detection - {city_name} ({title})", fontsize=16)
        plt.show()
    elif mode == "NBDI":
        plt.subplot(1, 5, 1)
        plt.title('Image avant (t1)')
        plt.imshow(image_t1, cmap='gray')
        plt.subplot(1, 5, 2)
        plt.title('Image après (t2)')
        plt.imshow(image_t2, cmap='gray')
        plt.subplot(1, 5, 3)
        plt.title('Différence normale')
        plt.imshow(diff_norm, cmap='gray')
        plt.subplot(1, 5, 4)
        plt.title('Carte de changement (Otsu)')
        plt.imshow(change_map, cmap='gray')
        plt.subplot(1, 5, 5)
        plt.title('Ground truth')
        plt.imshow(ground_truth, cmap='gray')
        plt.suptitle(f"Change detection - {city_name} ({title})", fontsize=16)
        plt.show()

def evaluate_change_map(change_map: np.ndarray, ground_truth: np.ndarray)->dict:
    """
    Évalue la carte de changement en calculant les métriques de performance.
    
    Args:
        change_map: Carte de changement (0->255)
        ground_truth: Ground truth (0 ou 1)
    """
    #On transforme la carte de changement (0->255) en une image binaire (0 ou 1)
    pred = (change_map > 0).astype(np.uint8)
    gt = (ground_truth > 0).astype(np.uint8)
    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))
    
    # Métriques
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # IoU (Intersection over Union) / Jaccard Index
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    
    # Kappa de Cohen (corrige le hasard)
    po = accuracy  # Accord observé
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / ((TP + TN + FP + FN) ** 2)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
    
    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'IoU': iou,
        'Kappa': kappa
    }

metrics_RGB = []
metrics_S1 = []
metrics_B11 = []
metrics_B11_mask_NBDI = []
metrics_B8 = []
metrics_CVA = []

for (
    (image_s1_t1, image_s1_t2),
    (image_s2_t1, image_s2_t2),
    (image_s2_b11_t1, image_s2_b11_t2),
    (image_s2_b8_t1, image_s2_b8_t2),
    (cva_t1, cva_t2),
    mask_build_up_index,
    ground_truth,
    city_name
) in tqdm(
    zip(
        images_s1,
        images_s2_RGB,
        images_s2_B11,
        images_s2_B8,
        CVAs,
        masks_build_up_index,
        labels,
        city_names
    ),
    total=len(city_names),
    desc="Processing cities"
): 
    diff_norm_s1, change_map_s1 = compute_change_map(image_s1_t1, image_s1_t2)
    diff_norm_s2_RGB, change_map_s2_RGB = compute_change_map(image_s2_t1, image_s2_t2)
    diff_norm_s2_b11, change_map_s2_b11 = compute_change_map(image_s2_b11_t1, image_s2_b11_t2)
    diff_norm_s2_b8, change_map_s2_b8 = compute_change_map(image_s2_b8_t1, image_s2_b8_t2)
    diff_norm_cva, change_map_cva = compute_change_map(cva_t1, cva_t2)
    change_map_s2_b11_mask_NBDI = diff_norm_s2_b11 * mask_build_up_index
    
    change_map_s2_b11_mask_NBDI = change_map_s2_b11_mask_NBDI.astype(np.uint8)

    change_map_s2_RGB, ground_truth = fitAtoB(change_map_s2_RGB, ground_truth)
    change_map_s2_b11_mask_NBDI, ground_truth = fitAtoB(change_map_s2_b11_mask_NBDI, ground_truth)
    change_map_s2_b8, ground_truth = fitAtoB(change_map_s2_b8, ground_truth)
    change_map_s2_b11, ground_truth = fitAtoB(change_map_s2_b11, ground_truth)
    change_map_s1, ground_truth = fitAtoB(change_map_s1, ground_truth)
    change_map_cva, ground_truth = fitAtoB(change_map_cva, ground_truth)

    metrics_RGB.append(evaluate_change_map(change_map_s2_RGB, ground_truth))
    metrics_S1.append(evaluate_change_map(change_map_s1, ground_truth))
    metrics_B11.append(evaluate_change_map(change_map_s2_b11, ground_truth))
    metrics_B11_mask_NBDI.append(evaluate_change_map(change_map_s2_b11_mask_NBDI, ground_truth))
    metrics_B8.append(evaluate_change_map(change_map_s2_b8, ground_truth))
    metrics_CVA.append(evaluate_change_map(change_map_cva, ground_truth))

    visualize_change_map(image_s2_t1, image_s2_t2, diff_norm_s2_RGB, change_map_s2_RGB, ground_truth, "RGB", city_name, title="RGB")
    visualize_change_map(image_s1_t1, image_s1_t2, diff_norm_s1, change_map_s1, ground_truth, "S1", city_name, title="S1")
    visualize_change_map(image_s2_b11_t1, image_s2_b11_t2, diff_norm_s2_b11, change_map_s2_b11, ground_truth, "B", city_name, title="B11")#4
    visualize_change_map(image_s2_b11_t1, image_s2_b11_t2, diff_norm_s2_b11, change_map_s2_b11_mask_NBDI, ground_truth, "B", city_name, title="B11 mask NBDI")#3
    visualize_change_map(image_s2_b8_t1, image_s2_b8_t2, diff_norm_s2_b8, change_map_s2_b8, ground_truth, "B", city_name, title="B8")#2
    visualize_change_map(cva_t1, cva_t2, diff_norm_cva, change_map_cva, ground_truth, "CVA", city_name, title="CVA")#1

def mean_metric(metrics_list, key):
    return np.round(np.mean([m[key] for m in metrics_list]), 4)

print(
    "Accuracy RGB: ", mean_metric(metrics_RGB, 'Accuracy'),
    "Precision RGB: ", mean_metric(metrics_RGB, 'Precision'),
    "Recall RGB: ", mean_metric(metrics_RGB, 'Recall'),
    "F1-Score RGB: ", mean_metric(metrics_RGB, 'F1-Score'),
    "IoU RGB: ", mean_metric(metrics_RGB, 'IoU'),
    "Kappa RGB: ", mean_metric(metrics_RGB, 'Kappa')
)
print(
    "Accuracy S1: ", mean_metric(metrics_S1, 'Accuracy'),
    "Precision S1: ", mean_metric(metrics_S1, 'Precision'),
    "Recall S1: ", mean_metric(metrics_S1, 'Recall'),
    "F1-Score S1: ", mean_metric(metrics_S1, 'F1-Score'),
    "IoU S1: ", mean_metric(metrics_S1, 'IoU'),
    "Kappa S1: ", mean_metric(metrics_S1, 'Kappa')
)
print(
    "Accuracy B11: ", mean_metric(metrics_B11, 'Accuracy'),
    "Precision B11: ", mean_metric(metrics_B11, 'Precision'),
    "Recall B11: ", mean_metric(metrics_B11, 'Recall'),
    "F1-Score B11: ", mean_metric(metrics_B11, 'F1-Score'),
    "IoU B11: ", mean_metric(metrics_B11, 'IoU'),
    "Kappa B11: ", mean_metric(metrics_B11, 'Kappa')
)
print(
    "Accuracy B11 mask NBDI: ", mean_metric(metrics_B11_mask_NBDI, 'Accuracy'),
    "Precision B11 mask NBDI: ", mean_metric(metrics_B11_mask_NBDI, 'Precision'),
    "Recall B11 mask NBDI: ", mean_metric(metrics_B11_mask_NBDI, 'Recall'),
    "F1-Score B11 mask NBDI: ", mean_metric(metrics_B11_mask_NBDI, 'F1-Score'),
    "IoU B11 mask NBDI: ", mean_metric(metrics_B11_mask_NBDI, 'IoU'),
    "Kappa B11 mask NBDI: ", mean_metric(metrics_B11_mask_NBDI, 'Kappa')
)
print(
    "Accuracy B8: ", mean_metric(metrics_B8, 'Accuracy'),
    "Precision B8: ", mean_metric(metrics_B8, 'Precision'),
    "Recall B8: ", mean_metric(metrics_B8, 'Recall'),
    "F1-Score B8: ", mean_metric(metrics_B8, 'F1-Score'),
    "IoU B8: ", mean_metric(metrics_B8, 'IoU'),
    "Kappa B8: ", mean_metric(metrics_B8, 'Kappa')
)
print(
    "Accuracy CVA: ", mean_metric(metrics_CVA, 'Accuracy'),
    "Precision CVA: ", mean_metric(metrics_CVA, 'Precision'),
    "Recall CVA: ", mean_metric(metrics_CVA, 'Recall'),
    "F1-Score CVA: ", mean_metric(metrics_CVA, 'F1-Score'),
    "IoU CVA: ", mean_metric(metrics_CVA, 'IoU'),
    "Kappa CVA: ", mean_metric(metrics_CVA, 'Kappa')
)