import os
import torch

root_dir = os.path.dirname(os.path.abspath(__file__))

config = {
	"image_dir_uid": os.path.join(root_dir, "UID 2021"),
	"annotations_file_uid": os.path.join(root_dir, "UID 2021/mos.xlsx"),
	"image_dir_saud": os.path.join(root_dir, "SAUD/Enhanced"),
	"annotations_file_saud": os.path.join(root_dir, "SAUD/mos.xlsx"),
	"features": {
		"msaeqa_uid": os.path.join(root_dir, "MSAEQA/features_uid.xlsx"),
		"msaeqa_saud": os.path.join(root_dir, "MSAEQA/features_saud.csv"),
	},
	"pretrained_models": {
		"scnn": os.path.join(root_dir, "DBCNN/Pretrained SCNN.pkl")
	},
	"trained_models": {
		"CNNIQA_UID": os.path.join(root_dir, "CNNIQA/CNNIQA_UID_model"),
		"CNNIQA_SAUD": os.path.join(root_dir, "CNNIQA/CNNIQA_SAUD_model"),
		"DBCNN_UID": os.path.join(root_dir, "DBCNN/DBCNN_UID_model"),
		"DBCNN_SAUD": os.path.join(root_dir, "DBCNN/DBCNN_SAUD_model"),
		"HyperIQA_UID": os.path.join(root_dir, "HyperIQA/HyperIQA_UID_model"),
		"HyperIQA_SAUD": os.path.join(root_dir, "HyperIQA/HyperIQA_SAUD_model"),
		"MAUIQA_UID": os.path.join(root_dir, "MAUIQA/MAUIQA_UID_model"),
		"MAUIQA_SAUD": os.path.join(root_dir, "MAUIQA/MAUIQA_SAUD_model"),
		"MSAEQA_UID": os.path.join(root_dir, "MSAEQA/MSAEQA_UID_model"),
		"MSAEQA_SAUD": os.path.join(root_dir, "MSAEQA/MSAEQA_SAUD_model"),
		"LRFusionIQA_UID": os.path.join(root_dir, "FusionIQA/LRFusionIQA_UID_model"),
		"LRFusionIQA_SAUD": os.path.join(root_dir, "FusionIQA/LRFusionIQA_SAUD_model"),
		"SVRFusionIQA_UID": os.path.join(root_dir, "FusionIQA/SVRFusionIQA_UID_model"),
		"SVRFusionIQA_SAUD": os.path.join(root_dir, "FusionIQA/SVRFusionIQA_SAUD_model"),
	}
}

def get_device():
	# Metal Performance Shaders (MPS) is a high-performance library that is used to accelerate deep learning training on Apple devices.
	if torch.backends.mps.is_available():
		return torch.device("mps")
	# CUDA is a parallel computing platform and application programming interface model created by Nvidia.
	if torch.cuda.is_available():
		return torch.device("cuda")
	# If no GPU is available, use the CPU
	return torch.device("cpu")
