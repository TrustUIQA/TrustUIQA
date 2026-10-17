import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from UIDDataset import create_datasets
from MAUIQA.model import MAUIQA
import torch
from torch import nn
from scipy.stats import spearmanr, pearsonr
import numpy as np
from config import config
import matplotlib.pyplot as plt

def test_model(model, test_loader):
		model.eval()
		SROCC = []
		PLCC = []
		RMSE = []
		all_predicted = []
		all_target = []
		with torch.no_grad():
				for images, labels in test_loader:
						images, labels = images.to(device), labels.to(device)
						
						outputs = model(images)
						predicted = outputs.squeeze().cpu().numpy()
						target = labels.squeeze().cpu().numpy()

						all_predicted.extend(predicted)
						all_target.extend(target)

						srocc, _ = spearmanr(predicted, target)
						plcc, _ = pearsonr(predicted, target)
						rmse = np.sqrt(np.mean((predicted-target)**2))
						SROCC.append(srocc)
						PLCC.append(plcc)
						RMSE.append(rmse)


		return sum(SROCC) / len(SROCC), sum(PLCC) / len(PLCC), sum(RMSE) / len(RMSE), all_predicted, all_target
				
	

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
	model = MAUIQA()
	model = model.to(device)
	model.load_state_dict(torch.load(config['trained_models']['MAUIQA_UID']))

	srocc, plcc, rmse, predicted_scores, mos_scores = test_model(model, test_loader)

	print(f"SROCC: {srocc}, PLCC: {plcc}, RMSE: {rmse}")
	
	plt.figure(figsize=(10, 6))
	plt.scatter(mos_scores, predicted_scores, color='blue', alpha=0.5)
	plt.plot([min(mos_scores), max(mos_scores)], [min(mos_scores), max(mos_scores)], color='red', linestyle='--')  # Line of equality
	plt.xlabel('MOS Scores')
	plt.ylabel('Predicted Scores')
	plt.title('MAUIQA Model Predicted Scores vs MOS Scores')
	plt.grid(True)
	
	# Save the plot as an image
	scatter_plot_path = os.path.join(os.path.dirname(__file__), 'predicted_vs_mos_scatter.png')
	plt.savefig(scatter_plot_path)
	plt.close()

	print(f"Scatter plot saved as: {scatter_plot_path}")
