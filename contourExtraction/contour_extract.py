import cv2
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def find_contours(image_path):
    pil_image = Image.open(image_path)
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def process_dataset(data_dir):
    data = []
    index_number = 0
    species_number = 67

    for species_name in os.listdir(data_dir):
        species_dir = os.path.join(data_dir, species_name)
        if os.path.isdir(species_dir):
            for image_file in os.listdir(species_dir):
                # print(image_file)
                if image_file[-3:] != "jpg":
                    continue
                image_path = os.path.join(species_dir, image_file)
                contour = find_contours(image_path)
                for p in sorted(contour,key = lambda x:len(x))[-1][::20,::20,:]:
                    # print(p)
                # Extract x and y coordinates
                    x, y = p.ravel()
                    data.append([index_number, image_file, species_number,species_name, x, y])
                index_number += 1
        species_number += 1

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['indexNumber', 'indexName', 'speicesNumber','speciesName', 'X', 'Y'])
    df.to_csv('leaf_dataset5.csv', index=False)

