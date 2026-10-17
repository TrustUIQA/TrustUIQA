import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from config import config
from scipy.stats import spearmanr, pearsonr
import joblib


def logistic_regression(q, lambda1, lambda2, lambda3, lambda4, lambda5):
    q_prime = lambda1 * (0.5 - 1 / (1 + np.exp(lambda2 * (q - lambda3)))) + lambda4 * q + lambda5
    return q_prime

def lg_param(y_pred,y_test):
    # Calculate the initial guesses for the parameters
    lambda1 = np.max(y_test)
    lambda2 = np.min(y_test)
    
    lambda3 = np.mean(y_pred)
    lambda4 = np.std(y_pred) / 4

    #lambda4 = 1
    
    lambda5 = 1

    q_prime = logistic_regression(y_pred, lambda1, lambda2, lambda3, lambda4, lambda5)

    return q_prime

if __name__ == "__main__":
                
		features_df = pd.read_excel(config['features']['msaeqa_uid'])
		if 'Name' in features_df.columns:
			features_df = features_df.drop(columns=['Name'])
		features = features_df.iloc[:, :].values
                
		labels_df = pd.read_excel(config['annotations_file_uid'])
		labels = labels_df.iloc[:, -1].values
                
		X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=3)
		svr_linear = SVR(kernel='rbf', C=1.0, epsilon=0.1)
		svr_linear.fit(X_train, y_train)
		joblib.dump(svr_linear, config['trained_models']['MSAEQA_UID'])
		y_pred = svr_linear.predict(X_test)
		q_prime = lg_param(y_pred, y_test)
		srocc, _ = spearmanr(q_prime, y_test)
		plcc, _ = pearsonr(q_prime,y_test)				
		rmse = np.sqrt(np.mean((q_prime-y_test)**2))
		

		print(f"SROCC: {srocc}, PLCC: {plcc}, RMSE: {rmse}")
                        
		


