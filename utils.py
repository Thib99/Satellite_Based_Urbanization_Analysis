import numpy as np
from pathlib import Path
import tifffile as tiff
import imagecodecs
import cv2
import matplotlib.pyplot as plt
########################################################
# DATA LOADING
########################################################
def get_images_from_folder_S1(folder_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:

    images = []
    print("Extracting images from folder S1...")
    # Parcourt chaque dossier de ville sous folder_path
    for city_dir in sorted([p for p in folder_path.iterdir() if p.is_dir()]):
        city_images = []
        for sub in ("imgs_1", "imgs_2"):
            subdir = city_dir / sub
            tif_path = sorted(subdir.glob("*.tif"))
            tif_path = tif_path[0]

            img_array = np.asarray(tiff.imread(str(tif_path)))
            city_images.append(img_array)
        images.append(city_images)
 
    print(f"Extracted {len(images)} pairs of images from the folder.")
    return images

def get_images_from_folder_S2(folder_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:

    images = []
    print("Extracting images from folder S2...")    
    for city_dir in sorted([p for p in folder_path.iterdir() if p.is_dir()]):
        subdir = city_dir / "pair"
        img_1 = np.asarray(cv2.imread( str(subdir / "img1.png")))
        img_2 = np.asarray(cv2.imread( str(subdir / "img2.png")))
        img_1_RGB = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        img_2_RGB = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        images.append((img_1_RGB, img_2_RGB))
 
    print(f"Extracted {len(images)} pairs of images from the folder.")
    return images

def get_B_from_folder_S2(folder_path: Path, band_number: str) -> list[np.ndarray]:
    band_images = []
    print(f"Extracting B{band_number} from folder S2...")
    for city_dir in sorted([p for p in folder_path.iterdir() if p.is_dir()]):
        city_images = []
        for sub in ("imgs_1", "imgs_2"):
            subdir = city_dir / sub
            tif_path = sorted(subdir.glob(f"*B{band_number}.tif"))
            tif_path = tif_path[0]

            img_array = np.asarray(tiff.imread(str(tif_path)))
            img_array_expanded = np.expand_dims(img_array, axis=2)
            city_images.append(img_array_expanded)
        band_images.append(city_images)
    print(f"Extracted {len(band_images)} pairs of images from the folder.")
    return band_images

def get_labels_from_folder(folder_path: Path) -> list[np.ndarray]:
    labels = []
    print("Extracting labels from folder...")
    for city_dir in sorted([p for p in folder_path.iterdir() if p.is_dir()]):
        subdir = city_dir / "cm"
        label = np.asarray(cv2.imread( str(subdir / "cm.png"), cv2.IMREAD_GRAYSCALE))
        labels.append(label)
    print(f"Extracted {len(labels)} labels from the folder.")
    return labels

def get_city_names_from_folder(folder_path: Path) -> list[str]:
    city_names = []
    for city_dir in sorted([p for p in folder_path.iterdir() if p.is_dir()]):
        city_names.append(city_dir.name)
    return city_names

########################################################
# DATA VISUALIZATION
########################################################
def visualize_data_sizes(images : list[tuple[np.ndarray, np.ndarray]]) -> bool:
    nb_images_different_sizes = 0
    for i in range(len(images)):
        if images[i][0].shape != images[i][1].shape:
            print(f"Image {i} has different sizes: {images[i][0].shape} and {images[i][1].shape}")
            nb_images_different_sizes += 1
    if nb_images_different_sizes == 0:
        print("All images have the same size")


def plot_one_image(image: np.ndarray, channels: int):
    if channels == 1:
        plt.imshow(image, cmap='gray')
    elif channels == 3:
        plt.imshow(image)
    plt.show()


########################################################
# DATA RESHAPING
########################################################
# Certaines paires d'imagesn'ont pas exactement la même taille. 3 solutions différentes pour faire face à ce problème :
# 1. Basic crop
# 2. Center crop
# 3. Resize to the smallest size

def basic_crop_to_match(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if img1.shape[:2] != img2.shape[:2]:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        
        if img1.ndim == 3:
            img1_cropped = img1[:h, :w, :]
        else:
            img1_cropped = img1[:h, :w]
            
        if img2.ndim == 3:
            img2_cropped = img2[:h, :w, :]
        else:
            img2_cropped = img2[:h, :w]
    
        return img1_cropped, img2_cropped
    else:
        return img1, img2

def center_crop_to_match(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    if img1.shape[:2] != img2.shape[:2]:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])

        h_off1 = (img1.shape[0] - h) // 2
        w_off1 = (img1.shape[1] - w) // 2
        h_off2 = (img2.shape[0] - h) // 2
        w_off2 = (img2.shape[1] - w) // 2

        # Gestion des images 2D et 3D
        if img1.ndim == 3:
            img1_c = img1[h_off1:h_off1 + h, w_off1:w_off1 + w, :]
        else:
            img1_c = img1[h_off1:h_off1 + h, w_off1:w_off1 + w]
            
        if img2.ndim == 3:
            img2_c = img2[h_off2:h_off2 + h, w_off2:w_off2 + w, :]
        else:
            img2_c = img2[h_off2:h_off2 + h, w_off2:w_off2 + w]
            
        return img1_c, img2_c
    else:
        return img1, img2

def resize_to_min(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if img1.shape != img2.shape:
        h_min = min(img1.shape[0], img2.shape[0])
        w_min = min(img1.shape[1], img2.shape[1])
        
        img1_resized = cv2.resize(img1, (w_min, h_min), interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2, (w_min, h_min), interpolation=cv2.INTER_AREA)
        
        return img1_resized, img2_resized
    else:
        return img1, img2


def upsample_images_A(imgsA: list[tuple[np.ndarray, np.ndarray]], imgsB: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Upsample les images de la liste A pour qu'elles aient la même taille que les images de la liste B.
    Args:
        imgsA: list of tuples of images (img1, img2) 
        imgsB: list of tuples of images (img1, img2)
    Returns:
        list of tuples of images (img1, img2)
    """
    upsampled_imgsA = []
    for (imgA1, imgA2), (imgB1, imgB2) in zip(imgsA, imgsB):
        imgA1, imgB1 = fitAtoB(imgA1, imgB1)
        imgA2, imgB2 = fitAtoB(imgA2, imgB2)
        imgA1 = np.expand_dims(imgA1, axis=2)
        imgA2 = np.expand_dims(imgA2, axis=2)
        upsampled_imgsA.append((imgA1, imgA2))
    return upsampled_imgsA

def fitAtoB(imgA: np.ndarray, imgB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Redimensionne l'image A pour qu'elle ait exactement la même taille que l'image B.
    
    Args:
        imgA: Image à redimensionner
        imgB: Image de référence
        
    Returns:
        Tuple contenant (imgA redimensionnée, imgB inchangée)
    """
    if imgA.shape[:2] != imgB.shape[:2]:
        h_target = imgB.shape[0]
        w_target = imgB.shape[1]
        
        # Choisir l'interpolation selon si on agrandi ou réduit
        if h_target < imgA.shape[0] or w_target < imgA.shape[1]:
            # Downsampling
            interpolation = cv2.INTER_AREA
        else:
            # Upsampling
            interpolation = cv2.INTER_LINEAR
        
        imgA_resized = cv2.resize(imgA, (w_target, h_target), interpolation=interpolation)
        
        return imgA_resized, imgB
    else:
        return imgA, imgB
            


def normalize_image(image: np.ndarray, val_max: int) -> np.ndarray:
    return image / val_max