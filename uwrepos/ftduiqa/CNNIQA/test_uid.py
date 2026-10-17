import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from UIDDataset import create_datasets
from CNNIQA.model import CNNIQAnet
import torch
from torch import nn
from scipy.stats import spearmanr, pearsonr
import numpy as np
from config import config

def test_model(model, test_loader):
		model.eval()
		SROCC = []
		PLCC = []
		RMSE = []
		with torch.no_grad():
				for images, labels in test_loader:
						images, labels = images.to(device), labels.to(device)
						
						outputs = model(images)
						predicted = outputs.squeeze().cpu().numpy()
						target = labels.squeeze().cpu().numpy()
						srocc, _ = spearmanr(predicted, target)
						plcc, _ = pearsonr(predicted, target)
						rmse = np.sqrt(np.mean((predicted-target)**2))
						SROCC.append(srocc)
						PLCC.append(plcc)
						RMSE.append(rmse)

		return sum(SROCC) / len(SROCC), sum(PLCC) / len(PLCC), sum(RMSE) / len(RMSE)
				
	

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Define the transform
	transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
	])

	# Create datasets
	train_dataset, val_dataset, test_dataset = create_datasets(
			annotations_file=config['annotations_file_uid'], 
			image_dir=config['image_dir_uid'], 
			test_size=0.2, 
			val_size=0.1, 
			transform=transform
	)

	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


	# Instantiate the model
	model = CNNIQAnet(ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800)
	model = model.to(device)
	model.load_state_dict(torch.load(config['trained_models']['CNNIQA_UID']))

	srocc, plcc, rmse = test_model(model, test_loader)

	print(f"SROCC: {srocc}, PLCC: {plcc}, RMSE: {rmse}")
