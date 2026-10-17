import torch
import numpy as np
from config import config
from CNNIQA.model import CNNIQAnet
from DBCNN.model import DBCNN
from HyperIQA.model import HyperNet, TargetNet
from MAUIQA.model import MAUIQA
from MSAEQA.feature_extractor import MSAEQA
import joblib
from scipy.stats import spearmanr, pearsonr




class FusionIQA:
	def __init__(self, device):
		self.device = device
		self.models = self.get_models()

	def get_models(self):
		cnn_iqa_model = CNNIQAnet(ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800)
		cnn_iqa_model = cnn_iqa_model.to(self.device)
		cnn_iqa_model.load_state_dict(torch.load(config['trained_models']['CNNIQA_UID']))

		scnn_root = config['pretrained_models']['scnn'] 
		dbcnn_model = DBCNN(scnn_root, {'fc': True})
		dbcnn_model = dbcnn_model.to(self.device)
		dbcnn_model.load_state_dict(torch.load(config['trained_models']['DBCNN_UID']))

		hyper_iqa_model = HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
		hyper_iqa_model = hyper_iqa_model.to(self.device)
		hyper_iqa_model.load_state_dict(torch.load(config['trained_models']['HyperIQA_UID']))

		multi_scale_attention_iqa_model = MAUIQA()
		multi_scale_attention_iqa_model = multi_scale_attention_iqa_model.to(self.device)
		multi_scale_attention_iqa_model.load_state_dict(torch.load(config['trained_models']['MAUIQA_UID']))


		# traditional modal
		multi_scale_antogonistic_energy_model = joblib.load(config['trained_models']['MSAEQA_UID'])

		return {
			'cnn_iqa_model': cnn_iqa_model,
			'dbcnn_model': dbcnn_model,
			'hyper_iqa_model': hyper_iqa_model,
			'multi_scale_attention_iqa_model': multi_scale_attention_iqa_model,
			'multi_scale_antogonistic_energy_model': multi_scale_antogonistic_energy_model
		}

	def predict_dl_model(self, model_name, model, image):
		model.eval()
		with torch.no_grad():
			if model_name == "hyper_iqa_model":
				paras = model(image)
				target_model = TargetNet(paras).to(self.device)
				output = target_model(paras['target_in_vec'])
				output = output.unsqueeze(1)
			else:
				output = model(image)
			return output	

	def extract_features(self, data_loader):
			scores = []
			ground_truth = []
			count = 1
			for images, labels in data_loader:
				print(f"Batch: {count}/{len(data_loader)}")
				count += 1
				images = images.to(self.device)
				batch_scores = []
				for model_name, model in self.models.items():
						if model_name == 'multi_scale_antogonistic_energy_model':
							feature_extractor = MSAEQA()
							output = []
							for i in range(images.size(0)):
									image = images[i].cpu().numpy()  # Add batch dimension
									features = feature_extractor.extract_features(image)
									features = np.array(features).reshape(-1, 198)  # Ensure features have shape (N, 198)
									score = model.predict(features)
									output.append(score)
							output = np.array(output).reshape(images.size(0), 1)
							batch_scores.append(output)
						else:
							output = self.predict_dl_model(model_name, model, images).cpu().numpy()
							batch_scores.append(output)
						predicted = output.squeeze()
						target = labels.cpu().numpy()
						srocc, _ = spearmanr(target, predicted)
						plcc, _ = pearsonr(target, predicted)
						rmse = np.sqrt(np.mean((predicted - target)**2))
						print(f"{model_name} -> SROCC: {srocc}, PLCC: {plcc}, RMSE: {rmse}")
				batch_scores = np.concatenate(batch_scores, axis=1)
				scores.append(batch_scores)
				ground_truth.append(labels.cpu().numpy())
			return np.vstack(scores), np.hstack(ground_truth)






	
