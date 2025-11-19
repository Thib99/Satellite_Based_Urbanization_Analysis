from ast import And
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from collections import defaultdict



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


# array of VV only : 
images_s1_VV = [(img1[:,:,0], img2[:,:,0]) for img1, img2 in images_s1]
# array of VH only : 
images_s1_VH = [(img1[:,:,1], img2[:,:,1]) for img1, img2 in images_s1]


# SAR channels : VV and VH
def compute_change_map_VVVH(img1: np.ndarray, img2: np.ndarray, alpha: float, beta: float)->np.ndarray:
    """
    Compute the change map for the VV and VH channels
    Args:
        img1: image before
        img2: image after
        alpha: scaling factor for the VV channel
        beta: scaling factor for the VH channel
    Returns:
        change map
    """
    VV_t1, VV_t2 = img1[:,:,0], img2[:,:,0]
    VH_t1, VH_t2 = img1[:,:,1], img2[:,:,1]
    dVV = np.abs(VV_t2.astype(np.float32) - VV_t1.astype(np.float32))
    dVH = np.abs(VH_t2.astype(np.float32) - VH_t1.astype(np.float32))
    scaled_dVV = alpha * dVV
    scaled_dVH = beta * dVH
    diff_gray = scaled_dVV + scaled_dVH

    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    diff_smooth = cv2.GaussianBlur(diff_norm, (5,5), 1)

    # Otsu cherche automatiquement le meilleur seuil pour séparer "inchangé" / "changé" dans la carte de changement
    _, change_map = cv2.threshold(diff_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return diff_norm, change_map

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


# PCA
def make_PCA_features(*image_lists: list[tuple[np.ndarray, np.ndarray]])->list[np.ndarray]:
    """
    Fait la PCA sur les images de chaque liste et renvoie la magnitude des 2 secondes composantes principales
    Args:
        image_lists: list of lists of tuples of images (img1, img2)
    Returns:
        list of PCA features, carte des changements
    """
    n_pairs = len(image_lists[0])
    if not all(len(img_list) == n_pairs for img_list in image_lists):
        raise ValueError("Toutes les listes d'images doivent avoir la même longueur")
    
    PCA_features = []
    
    # on itère sur chaque paire d'images
    for i in range(n_pairs):
        # on extrait toutes les paires pour cet index
        pairs = [img_list[i] for img_list in image_lists]

        pairs = np.squeeze(pairs)
        
        # on sépare t1 et t2 pour chaque bande
        bands_t1 = [pair[0] for pair in pairs]
        bands_t2 = [pair[1] for pair in pairs]
        
        # on empile par pixel
        X1 = np.stack(bands_t1, axis=-1)  # shape (H, W, n_bands)
        X2 = np.stack(bands_t2, axis=-1)  # shape (H, W, n_bands)

        # on concatène temporelement
        X = np.concatenate([X1, X2], axis=-1)  # shape (H, W, 2*n_bands)

        # on met en vecteur (flatten)
        H, W, C = X.shape
        X_flat = X.reshape(-1, C)  # shape (N, 2*n_bands) où N = H*W
        
        # PCA
        pca = PCA(n_components=C)
        X_pca = pca.fit_transform(X_flat)  # shape (N, n_components)
        
        PC2 = X_pca[:, 1].reshape(H, W) # de dim H * W * 1
        PC3 = X_pca[:, 2].reshape(H, W) # de dim H * W * 1
        
        M = np.sqrt(PC2**2 + PC3**2)  # magnitude, de dim H * W * 1
        M_norm = (M - M.min()) / (M.max() - M.min())
        M_255 = (M_norm * 255).astype(np.uint8)

        _, binary = cv2.threshold(M_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        PCA_features.append(binary)
    return PCA_features


PCA_features = make_PCA_features(images_s2_B4, images_s2_B11, images_s2_B12)


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
        plt.title('Image before (t1)')
        plt.imshow(image_t1, cmap='jet')
        plt.subplot(1, 5, 2)
        plt.title('Image after (t2)')
        plt.imshow(image_t2, cmap='jet')
        plt.subplot(1, 5, 3)
        plt.title('Difference')
        plt.imshow(diff_norm, cmap='gray')
        plt.subplot(1, 5, 4)
        plt.title('Change map (Otsu)')
        plt.imshow(change_map, cmap='gray')
        plt.subplot(1, 5, 5)
        plt.title('Ground truth')
        plt.imshow(ground_truth, cmap='gray')
        plt.suptitle(f"Change detection - {city_name} ({title})", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        plt.show()
    elif mode == "B":
        plt.subplot(1, 5, 1)
        plt.title('Image before (t1)')
        plt.imshow(image_t1, cmap='gray')
        plt.subplot(1, 5, 2)
        plt.title('Image after (t2)')
        plt.imshow(image_t2, cmap='gray')
        plt.subplot(1, 5, 3)    
        plt.title('Difference')
        plt.imshow(diff_norm, cmap='gray')
        plt.subplot(1, 5, 4)
        plt.title('Change map (Otsu)')
        plt.imshow(change_map, cmap='gray')
        plt.subplot(1, 5, 5)
        plt.title('Ground truth')
        plt.imshow(ground_truth, cmap='gray')
        plt.suptitle(f"Change detection - {city_name} ({title})", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        plt.show()
    elif mode == "S1":
        plt.subplot(1, 5, 1)
        plt.title('Image before (t1)')
        plt.imshow(np.mean(image_t1, axis=-1), cmap='gray')
        plt.subplot(1, 5, 2)
        plt.title('Image after (t2)')
        plt.imshow(np.mean(image_t2, axis=-1), cmap='gray')
        plt.subplot(1, 5, 3)    
        plt.title('Difference')
        plt.imshow(diff_norm, cmap='gray')
        plt.subplot(1, 5, 4)
        plt.title('Change map (Otsu)')
        plt.imshow(change_map, cmap='gray')
        plt.subplot(1, 5, 5)
        plt.title('Ground truth')
        plt.imshow(ground_truth, cmap='gray')
        plt.suptitle(f"Change detection - {city_name} ({title})", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        plt.show()
    elif mode == "CVA":
        plt.subplot(1, 5, 1)
        plt.title('Image before (t1)')
        plt.imshow(image_t1, cmap='jet')
        plt.subplot(1, 5, 2)
        plt.title('Image after (t2)')
        plt.imshow(image_t2, cmap='jet')
        plt.subplot(1, 5, 3)
        plt.title('Difference')
        plt.imshow(diff_norm, cmap='gray')
        plt.subplot(1, 5, 4)
        plt.title('Change map (Otsu)')
        plt.imshow(change_map, cmap='gray')
        plt.subplot(1, 5, 5)
        plt.title('Ground truth')
        plt.imshow(ground_truth, cmap='gray')
        plt.suptitle(f"Change detection - {city_name} ({title})", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        plt.show()
    elif mode == "NBDI":
        plt.subplot(1, 5, 1)
        plt.title('Image before (t1)')
        plt.imshow(image_t1, cmap='gray')
        plt.subplot(1, 5, 2)
        plt.title('Image after (t2)')
        plt.imshow(image_t2, cmap='gray')
        plt.subplot(1, 5, 3)
        plt.title('Difference')
        plt.imshow(diff_norm, cmap='gray')
        plt.subplot(1, 5, 4)
        plt.title('Change map (Otsu)')
        plt.imshow(change_map, cmap='gray')
        plt.subplot(1, 5, 5)
        plt.title('Ground truth')
        plt.imshow(ground_truth, cmap='gray')
        plt.suptitle(f"Change detection - {city_name} ({title})", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        plt.show()
    elif mode == "PCA":
        plt.subplot(1, 4, 1)
        plt.title('Image before (t1)')
        plt.imshow(image_t1, cmap='jet')
        plt.subplot(1, 4, 2)
        plt.title('Image after (t2)')
        plt.imshow(image_t2, cmap='jet')
        plt.subplot(1, 4, 3)
        plt.title('Change map (Otsu)')
        plt.imshow(change_map, cmap='gray')
        plt.subplot(1, 4, 4)
        plt.title('Ground truth')
        plt.imshow(ground_truth, cmap='gray')
        plt.suptitle(f"Change detection - {city_name} ({title})", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
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

all_metrics = []
metrics_RGB = []
metrics_S1 = []
metrics_B11 = []
metrics_B11_mask_NBDI = []
metrics_B8 = []
metrics_CVA = []
metrics_PCA = []
metrics_S1_VVVH = []
metrics_S1_VV = []
metrics_S1_VH = []
for (
    (image_s1_t1, image_s1_t2),
    (image_s1_VV_t1, image_s1_VV_t2),
    (image_s1_VH_t1, image_s1_VH_t2),
    (image_s2_t1, image_s2_t2),
    (image_s2_b11_t1, image_s2_b11_t2),
    (image_s2_b8_t1, image_s2_b8_t2),
    (cva_t1, cva_t2),
    mask_build_up_index,
    PCA_feature,
    ground_truth,
    city_name
) in tqdm(
    zip(
        images_s1,
        images_s1_VV,
        images_s1_VH,
        images_s2_RGB,
        images_s2_B11,
        images_s2_B8,
        CVAs,
        masks_build_up_index,
        PCA_features,
        labels,
        city_names
    ),
    total=len(city_names),
    desc="Processing cities"
): 
    diff_norm_s1, change_map_s1 = compute_change_map(image_s1_t1, image_s1_t2)
    diff_norm_s1_VV, change_map_s1_VV = compute_change_map(image_s1_VV_t1, image_s1_VV_t2)
    diff_norm_s1_VH, change_map_s1_VH = compute_change_map(image_s1_VH_t1, image_s1_VH_t2)
    diff_norm_s1_VVVH, change_map_s1_VVVH = compute_change_map_VVVH(image_s1_t1, image_s1_t2, 0.6, 0.4)
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
    change_map_cva, ground_truth = fitAtoB(change_map_cva, ground_truth)
    change_map_PCA, ground_truth = fitAtoB(PCA_feature, ground_truth)
    change_map_s1_VVVH, ground_truth = fitAtoB(change_map_s1_VVVH, ground_truth)
    change_map_s1, ground_truth = fitAtoB(change_map_s1, ground_truth)
    change_map_s1_VV, ground_truth = fitAtoB(change_map_s1_VV, ground_truth)
    change_map_s1_VH, ground_truth = fitAtoB(change_map_s1_VH, ground_truth)

    metrics_RGB.append(evaluate_change_map(change_map_s2_RGB, ground_truth))
    metrics_B11.append(evaluate_change_map(change_map_s2_b11, ground_truth))
    metrics_B11_mask_NBDI.append(evaluate_change_map(change_map_s2_b11_mask_NBDI, ground_truth))
    metrics_B8.append(evaluate_change_map(change_map_s2_b8, ground_truth))
    metrics_CVA.append(evaluate_change_map(change_map_cva, ground_truth))
    metrics_PCA.append(evaluate_change_map(change_map_PCA, ground_truth))
    metrics_S1_VVVH.append(evaluate_change_map(change_map_s1_VVVH, ground_truth))
    metrics_S1.append(evaluate_change_map(change_map_s1, ground_truth))
    metrics_S1_VV.append(evaluate_change_map(change_map_s1_VV, ground_truth))
    metrics_S1_VH.append(evaluate_change_map(change_map_s1_VH, ground_truth))
    all_metrics.append(metrics_RGB)
    all_metrics.append(metrics_B11)
    all_metrics.append(metrics_B11_mask_NBDI)
    all_metrics.append(metrics_B8)
    all_metrics.append(metrics_CVA)
    all_metrics.append(metrics_PCA)
    all_metrics.append(metrics_S1_VVVH)
    all_metrics.append(metrics_S1)
    all_metrics.append(metrics_S1_VV)
    all_metrics.append(metrics_S1_VH)

    """visualize_change_map(image_s2_t1, image_s2_t2, diff_norm_s2_RGB, change_map_s2_RGB, ground_truth, "RGB", city_name, title="RGB")
    visualize_change_map(image_s2_b11_t1, image_s2_b11_t2, diff_norm_s2_b11, change_map_s2_b11, ground_truth, "B", city_name, title="B11")
    visualize_change_map(image_s2_b11_t1, image_s2_b11_t2, diff_norm_s2_b11, change_map_s2_b11_mask_NBDI, ground_truth, "B", city_name, title="B11 mask NBDI")
    visualize_change_map(image_s2_b8_t1, image_s2_b8_t2, diff_norm_s2_b8, change_map_s2_b8, ground_truth, "B", city_name, title="B8")
    visualize_change_map(cva_t1, cva_t2, diff_norm_cva, change_map_cva, ground_truth, "CVA", city_name, title="CVA")
    visualize_change_map(image_s2_t1, image_s2_t2, None, change_map_PCA, ground_truth, "PCA", city_name, title="PCA")
    visualize_change_map(image_s1_t1, image_s1_t2, diff_norm_s1_VVVH, change_map_s1_VVVH, ground_truth, "S1", city_name, title="S1 VVVH")
    visualize_change_map(image_s1_t1, image_s1_t2, diff_norm_s1, change_map_s1, ground_truth, "S1", city_name, title="S1")
    visualize_change_map(image_s1_VV_t1, image_s1_VV_t2, diff_norm_s1_VV, change_map_s1_VV, ground_truth, "B", city_name, title="S1 VV")
    visualize_change_map(image_s1_VH_t1, image_s1_VH_t2, diff_norm_s1_VH, change_map_s1_VH, ground_truth, "B", city_name, title="S1 VH")"""



########################################################
# Evaluation des méthodes
########################################################
def test_alpha_beta(images_s1 : list[tuple[np.ndarray, np.ndarray]], ground_truth : list[np.ndarray])->list[dict]:
    metrics_S1_VVVH = []
    alpha_beta_list = []
    for (image_s1_t1, image_s1_t2), ground_truth in zip(images_s1, ground_truth):
        for alpha in np.arange(0, 1, 0.1):
            for beta in np.arange(0, 1, 0.1):
                diff_norm_s1_VVVH, change_map_s1_VVVH = compute_change_map_VVVH(image_s1_t1, image_s1_t2, alpha, beta)
                change_map_s1_VVVH, ground_truth = fitAtoB(change_map_s1_VVVH, ground_truth)
                metrics_S1_VVVH.append(evaluate_change_map(change_map_s1_VVVH, ground_truth))
                alpha_beta_list.append((alpha, beta))
    return metrics_S1_VVVH, alpha_beta_list 

metrics_S1_VVVH, alpha_beta_list = test_alpha_beta(images_s1, labels)

def plot_f1_alpha_beta_average(metrics_list, alpha_beta_list):
    """
    Affiche une heatmap du F1-score MOYEN pour chaque paire (alpha, beta)
    même si plusieurs images ont été testées.
    """

    # Regrouper les F1-score par (alpha, beta)
    grouped = defaultdict(list)
    for (alpha, beta), metrics in zip(alpha_beta_list, metrics_list):
        grouped[(alpha, beta)].append(metrics["F1-Score"])

    # Extraire toutes les valeurs possibles
    alphas = sorted(list(set([ab[0] for ab in alpha_beta_list])))
    betas  = sorted(list(set([ab[1] for ab in alpha_beta_list])))

    # Créer la grille
    f1_grid = np.zeros((len(alphas), len(betas)))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            f1_values = grouped[(alpha, beta)]
            f1_grid[i, j] = np.mean(f1_values)  # MOYENNE !

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(f1_grid, origin='lower', cmap='viridis',
               extent=[min(betas), max(betas), min(alphas), max(alphas)])
    plt.colorbar(label="F1 mean score")
    plt.xlabel("Beta")
    plt.ylabel("Alpha")
    plt.title("F1 mean score as a function of alpha and beta")
    plt.xticks(betas)
    plt.yticks(alphas)
    plt.show()

    return f1_grid, alphas, betas

f1_grid, alphas, betas = plot_f1_alpha_beta_average(metrics_S1_VVVH, alpha_beta_list)

def summarize_metric(metrics_list, key):
    """
    Retourne moyenne, écart-type, minimum et maximum pour une métrique donnée.
    """
    values = np.array([m[key] for m in metrics_list], dtype=float)

    return {
        "mean": np.round(np.mean(values), 4),
        "std": np.round(np.std(values), 4),
        "min": np.round(np.min(values), 4),
        "max": np.round(np.max(values), 4)
    }
def print_summary_metrics(metrics_list, prefix=""):
    metrics_names = ['F1-Score', 'IoU', 'Kappa']

    print(f"\nRésumé des métriques {prefix}:")
    for key in metrics_names:
        stats = summarize_metric(metrics_list, key)
        print(
            f"{key}: "
            f"mean={stats['mean']}, "
            f"std={stats['std']}, "
            f"min={stats['min']}, "
            f"max={stats['max']}"
        )
method_names = [
    "RGB", "B11", "B11_mask_NBDI", "B8", "CVA", "PCA", "S1_VVVH", "S1", "S1_VV", "S1_VH"
]
for name, metrics in zip(method_names, all_metrics):
    print_summary_metrics(metrics, prefix=name)