import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
from config import config
import cv2
from MSAEQA.feature_extractor import MSAEQA
import pandas as pd

def get_all_image_files(main_dataset_path):
		image_files = []
		for root, dirs, files in os.walk(main_dataset_path):
				for file in files:
						if file.endswith(('.png', '.jpg', '.jpeg')):
								last_two_folders = os.path.join(os.path.basename(os.path.dirname(root)), os.path.basename(root))
								id = os.path.join(last_two_folders, file)
								image_files.append({"id": id, "path": os.path.join(root, file)})
		return image_files

def load_images(image_files):
		images = []
		for img_data in image_files:
				img = cv2.imread(img_data['path'])
				if img is not None:
						images.append({"id": img_data['id'], "image": img})
		return images


if __name__ == "__main__":
	main_dataset_path = config['image_dir_saud']
	image_files = get_all_image_files(main_dataset_path)
	images = load_images(image_files)
	feature_extractor = MSAEQA()
	features = []
	img_count = 1
	for img_data in images:
		print(f"Extracting features: {img_count}/{len(images)}")
		feature = feature_extractor.extract_features(img_data['image'])
		feature_with_id = {"id": img_data['id'], **{f"feature_{i}": f for i, f in enumerate(feature)}}
		features.append(feature_with_id)
		img_count += 1
	df = pd.DataFrame(features)
	file_path = config['features']['msaeqa_saud']
	df.to_csv(file_path, index=False)
	print(f"Data has been written to {file_path}")

	