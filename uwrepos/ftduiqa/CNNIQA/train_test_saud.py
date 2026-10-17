import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from SAUDDataset import create_datasets
from CNNIQA.model import CNNIQAnet
import torch
from torch import nn
from scipy.stats import spearmanr, pearsonr
import numpy as np
from config import config, get_device






def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20):
		model.train()
		
		for epoch in range(epochs):
				running_loss = 0.0
				
				# Training phase
				for images, labels in train_loader:
						images, labels = images.to(device), labels.to(device)
						
						optimizer.zero_grad()
						
						outputs = model(images)
						loss = criterion(outputs, labels.unsqueeze(1))  # Unsqueeze labels to match the output size
						loss.backward()
						optimizer.step()
						
						running_loss += loss.item()
				
				# Validation phase
				model.eval()
				val_loss = 0.0
				with torch.no_grad():
						for images, labels in val_loader:
								images, labels = images.to(device), labels.to(device)
								
								outputs = model(images)
								loss = criterion(outputs, labels.unsqueeze(1))
								val_loss += loss.item()
				
				print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")
				
		return model

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
						rmse = np.sqrt(np.mean(predicted-target)**2)
						SROCC.append(srocc)
						PLCC.append(plcc)
						RMSE.append(rmse)

		return sum(SROCC) / len(SROCC), sum(PLCC) / len(PLCC), sum(RMSE) / len(RMSE)
				
if __name__ == "__main__":
	device = get_device()
	
	# Define the transform

	transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.366, 0.424, 0.413], std=[0.442, 0.484, 0.477])  
	])

	# Create datasets
	train_dataset, val_dataset, test_dataset = create_datasets(
			annotations_file=config['annotations_file_saud'], 
			image_dir=config['image_dir_saud'], 
			test_size=0.2, 
			val_size=0.1, 
			transform=transform
	)

	# Create DataLoaders
	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


	# Instantiate the model
	model = CNNIQAnet(ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800)
	model = model.to(device)

	# Loss function and optimizer
	criterion = nn.L1Loss()  # L1 loss for IQA
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

	# Train the model
	model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50)

	# Save the model
	torch.save(model.state_dict(), config['trained_models']['CNNIQA_SAUD'])

	# Test the model
	srocc, plcc, rmse = test_model(model, test_loader)

	print(f"SROCC: {srocc}, PLCC: {plcc}, RMSE: {rmse}")