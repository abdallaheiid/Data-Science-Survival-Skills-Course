# Load necessary libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

N = 4

def read_images(folder_path : str, num_images: int, meta_attribute: str) -> None:
    
    """_summary_
    """
    
    # Get how many files in the folder
    
    files_count = len([entry for entry in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, entry))]) // 3
    
    print("files count:" + str(files_count))
    plt.figure(figsize=(10,10))

    for img in range(num_images):
        img_id = np.random.choice(list(range(files_count)))
        print(img_id)
        
        file_path = folder_path + "/" + str(img_id)
        
        random_image = plt.imread(file_path + ".png")
        random_image_mask = plt.imread(file_path + "_seg.png")
        random_image_meta = json.load(open(file_path + ".meta"))
        
        print("random image shape: " + str(random_image.shape))
        print("random image mask shape: " + str(random_image_mask.shape))
        
        disorder_status = random_image_meta[meta_attribute]
        
        # masked image
        masked_image = np.ma.masked_where(random_image_mask == 1, np.mean(random_image, axis=2))
        
        # using subplot to show the 4 images in one figure ith overlaied mask
        plt.subplot(2,2,img+1)
        plt.imshow(np.mean(random_image, axis=2))
        plt.imshow(masked_image)
        plt.title(disorder_status)
        plt.axis('off')
        

        

read_images(folder_path="Mini_BAGLS_dataset", num_images=4, meta_attribute="Subject disorder status")


# Load the leaves.jpg image
leaves = plt.imread("leaves.jpg")

# Convert the image to grayscale
leaves_gray_light = np.max(leaves,axis=-1,keepdims=1) / 2 + np.min(leaves,axis=-1,keepdims=1) / 2

leaves_gray_avg = np.mean(leaves,axis=-1,keepdims=1)

weights = [0.2989, 0.5870, 0.1140]
leaves_lumin = np.sum(leaves * weights, axis=-1, keepdims=1)

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(leaves)
plt.title("Original image")
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(leaves_gray_light, cmap='gray')
plt.title("Lightness")
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(leaves_gray_avg, cmap='gray')
plt.title("Average")
plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(leaves_lumin, cmap='gray', alpha=0.9)
plt.title("Luminosity")
plt.axis('off')
plt.show()

    




