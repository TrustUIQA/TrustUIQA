import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader
from torchvision import transforms
from UIDDataset import create_datasets
import torch
from scipy.stats import spearmanr, pearsonr
import numpy as np
from config import config
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from FusionIQA.model import FusionIQA

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Define the transform
	transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
	])

	# Create datasets
	train_dataset, val_dataset, test_dataset  = create_datasets(
			annotations_file=config['annotations_file_uid'], 
			image_dir=config['image_dir_uid'], 
			test_size=0.2, 
			val_size=0.1, 
			transform=transform
	)

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


	model = FusionIQA(device)

	print("Training")
	features, labels = model.extract_features(train_loader)
	print("Testing")
	features, labels = model.extract_features(test_loader)

	print("Linear Regression")
	reg = LinearRegression()
	reg.fit(features, labels)
	joblib.dump(reg, config['trained_models']['LRFusionIQA_UID'])
	predictions = reg.predict(features)
	srocc, _ = spearmanr(labels, predictions)
	plcc, _ = pearsonr(labels, predictions)
	rmse = np.sqrt(np.mean((predictions - labels)**2))
	print(f"SROCC: {srocc}, PLCC: {plcc}, RMSE: {rmse}")

	print("SVR")
	svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
	svr.fit(features, labels)
	joblib.dump(svr, config['trained_models']['SVRFusionIQA_UID'])
	predictions = svr.predict(features)
	srocc, _ = spearmanr(labels, predictions)
	plcc, _ = pearsonr(labels, predictions)
	rmse = np.sqrt(np.mean((predictions - labels)**2))
	print(f"SROCC: {srocc}, PLCC: {plcc}, RMSE: {rmse}")





	
