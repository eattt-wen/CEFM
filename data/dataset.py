import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


#Dataset
class MelanomaDataset(Dataset):

    def __init__(self, image_dir, clinical_data_path, transform=None):
        self.image_dir = image_dir
        self.clinical_data = pd.read_csv(clinical_data_path)
        self.transform = transform

        self.clinical_data['image_name'] = self.clinical_data['image_name'].apply(lambda x: os.path.basename(x))
        self.clinical_data['image_name'] = self.clinical_data['image_name'].apply(self._force_add_extension)

        self._filter_valid_images()
        self.image_files = []
        self.clinical_features = []
        self.labels = []

        self._build_data_lists()
        print(f"Loaded {len(self.image_files)} valid samples")

    def _force_add_extension(self, image_name):
        file_name, file_extension = os.path.splitext(image_name)
        if file_extension == '':
            image_name = file_name + '.jpg'
        return image_name

    def _filter_valid_images(self):
        valid_indices = []
        for idx, row in self.clinical_data.iterrows():
            image_name = row['image_name']
            file_name, file_extension = os.path.splitext(image_name)
            file_extension = file_extension.lower()

            if file_extension == '' or file_extension not in ['.jpg', '.jpeg', '.png']:
                continue

            image_path = os.path.join(self.image_dir, image_name).replace("\\", "/")
            if os.path.exists(image_path):
                valid_indices.append(idx)

        self.clinical_data = self.clinical_data.loc[valid_indices].reset_index(drop=True)
        print(f"Filtered dataset size: {len(self.clinical_data)}")

        if len(self.clinical_data) == 0:
            raise ValueError("The dataset is empty. Please check if the image files exist or if the path is correct.")

    def _build_data_lists(self):
        for idx, row in self.clinical_data.iterrows():
            img_name = row['image_name']
            img_path = os.path.join(self.image_dir, img_name).replace("\\", "/")

            try:
                features = row.drop(['image_name', 'target']).values
                features = features.astype(np.float32)
                label = int(row['target'])

                self.image_files.append(img_path)
                self.clinical_features.append(features)
                self.labels.append(label)
            except Exception as e:
                print(f"Error processing sample {img_name}: {e}")
                continue

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_files[idx]).convert('RGB')

            if self.transform:
                image = self.transform(image)

            clinical_feature = self.clinical_features[idx]

            if isinstance(clinical_feature, np.ndarray) and clinical_feature.dtype == np.dtype('O'):
                clinical_feature = np.array([float(x) if x is not None else 0.0 for x in clinical_feature],
                                            dtype=np.float32)

            clinical_feature = torch.tensor(clinical_feature, dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.long)

            return image, clinical_feature, label
        except Exception as e:
            print(f"Error loading sample {self.image_files[idx]}: {e}")
            raise