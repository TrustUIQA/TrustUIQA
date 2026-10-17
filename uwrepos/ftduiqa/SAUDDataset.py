import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split


class SAUDDataset(Dataset):
		def __init__(self, img_paths, img_labels, transform=None):
				self.transform = transform
				self.img_paths = img_paths
				self.img_labels = img_labels

		def __len__(self):
				return len(self.img_paths)

		def __getitem__(self, idx):
				img_path = self.img_paths[idx]
				image = cv2.imread(img_path)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				image = Image.fromarray(image)
				
				if self.transform:
						image = self.transform(image)
				
				label = torch.tensor(self.img_labels[idx], dtype=torch.float32)
				return image, label


def create_datasets(annotations_file, image_dir, test_size=0.2, val_size=0.1, transform=None):
	# Read annotations file
	annotations_df = pd.read_excel(annotations_file)
	
	img_paths = []
	img_labels = []
	
	# Collect image paths and labels
	for root, _, files in os.walk(image_dir):
		for file in files:
			if file.endswith(('.png', '.jpg', '.jpeg')):
				img_paths.append(os.path.join(root, file))
				last_two_folders = os.path.join(os.path.basename(os.path.dirname(root)), os.path.basename(root))
				id = os.path.join(last_two_folders, file)
				mos_value = annotations_df.loc[annotations_df['Name'] == id, 'MOS'].values
				if len(mos_value) > 0:
					img_labels.append(mos_value[0])
				else:
					img_labels.append(0.0)  # Dummy value if MOS is missing

	# Split into training, validation, and test sets
	img_paths_train, img_paths_test, labels_train, labels_test = train_test_split(
		img_paths, img_labels, test_size=test_size, random_state=42)
	
	img_paths_train, img_paths_val, labels_train, labels_val = train_test_split(
		img_paths_train, labels_train, test_size=val_size, random_state=42)
	
	# Create datasets
	train_dataset = SAUDDataset(img_paths_train, labels_train, transform=transform)
	val_dataset = SAUDDataset(img_paths_val, labels_val, transform=transform)
	test_dataset = SAUDDataset(img_paths_test, labels_test, transform=transform)
	
	return train_dataset, val_dataset, test_dataset