import numpy as np
import cv2
import scipy
from scipy.special import gamma
import math
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from scipy.stats import weibull_min
import torch


def create_gaussian_pyramid(image, levels: int):
	gaussian_pyramid = [image]
	temp = image.copy()
	for i in range(levels - 1):
		temp = cv2.pyrDown(temp)
		gaussian_pyramid.append(temp)
	return gaussian_pyramid

def create_laplacian_pyramid(image):
	laplacian_pyramid = []
	for i in range(len(image) - 1):
		size = (image[i].shape[1], image[i].shape[0])
		expanded = cv2.pyrUp(image[i + 1], dstsize=size)
		laplacian = cv2.subtract(image[i], expanded)
		laplacian_pyramid.append(laplacian)
	laplacian_pyramid.append(image[-1])
	return laplacian_pyramid

class StatisticalFeatures():
	def get_max_chromaticity(self, image, levels: int):
		values = []
		for level in range(levels):
			b, g, r = cv2.split(image[level])
			max_gb  = np.maximum(g, b)
			max_chrominance = np.abs(r - max_gb)
			values.append(max_chrominance)
		return values
	
	def generate_gaussian_window(self, lw, sigma):
		sd = np.float32(sigma)
		lw = int(lw)
		weights = [0.0] * (2 * lw + 1)
		weights[lw] = 1.0
		sum = 1.0
		sd *= sd
		for ii in range(1, lw + 1):
				tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
				weights[lw + ii] = tmp
				weights[lw - ii] = tmp
				sum += 2.0 * tmp
		for ii in range(2 * lw + 1):
				weights[ii] /= sum
		return weights
	
	def compute_mscn_transform(self, values, C=1, avg_window=None, extend_mode='constant'):
		mscns = []
		if avg_window is None:
			avg_window = self.generate_gaussian_window(3, 7.0/6.0)
		for image in values:
			assert len(np.shape(image)) == 2
			h, w = np.shape(image)
			mu_image = np.zeros((h, w), dtype=np.float32)
			var_image = np.zeros((h, w), dtype=np.float32)
			image = np.array(image).astype(np.float32)
			scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
			scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
			scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
			scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
			var_image = np.sqrt(np.abs(var_image - mu_image**2))
			mscn = (image - mu_image) / (var_image + C)
			mscns.append(mscn)
		return mscns
	
	def estimate_ggd_param(self, value):
		gam = np.arange(0.2, 10.0, 0.001)
		r_gam = (gamma(1.0/gam)*gamma(3.0/gam))/((gamma(2.0/gam))**2)
		sigma_sq = np.mean(np.array(value)**2)  # Convert to NumPy array
		E = np.mean(np.abs(value))
		rho = sigma_sq / (E**2)
		
		r_gam = (gamma(1.0 / gam) * gamma(3.0 / gam)) / ((gamma(2.0 / gam)) ** 2)
		array_position = np.argmin((r_gam - rho) ** 2)
		alpha = gam[array_position]
		sigma = np.sqrt(sigma_sq)
		return alpha, sigma
	
	def compute_ggd(self, values):
		ggd = []
		for value in values:
			alpha_m, sigma = self.estimate_ggd_param(value)
			ggd.append(alpha_m)
			ggd.append(sigma)
		return ggd
	
	def paired_product(self, value):
		shift1 = np.roll(value.copy(), 1, axis=1)
		shift2 = np.roll(value.copy(), 1, axis=0)
		shift3 = np.roll(np.roll(value.copy(), 1, axis=0), 1, axis=1)
		shift4 = np.roll(np.roll(value.copy(), 1, axis=0), -1, axis=1)

		H_img = shift1 * value
		V_img = shift2 * value
		D1_img = shift3 * value
		D2_img = shift4 * value

		return H_img, V_img, D1_img, D2_img

	def get_aggd_features(self, imdata):
			# Flatten imdata
			imdata.shape = (len(imdata.flat),)
			imdata2 = imdata*imdata
			left_data = imdata2[imdata < 0]
			right_data = imdata2[imdata >= 0]
			left_mean_sqrt = 0
			right_mean_sqrt = 0
			if len(left_data) > 0:
					left_mean_sqrt = np.sqrt(np.average(left_data))
			if len(right_data) > 0:
					right_mean_sqrt = np.sqrt(np.average(right_data))

			if right_mean_sqrt != 0:
					gamma_hat = left_mean_sqrt/right_mean_sqrt
			else:
					gamma_hat = np.inf

			# Solve r-hat norm
			imdata2_mean = np.mean(imdata2)
			if imdata2_mean != 0:
					r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
			else:
					r_hat = np.inf
			rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1)*(gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

			# win = np.array(gen_gauss_window(3, 7.0/6.0))

			gamma_range = np.arange(0.2, 10, 0.001)
			a = scipy.special.gamma(2.0/gamma_range)
			a *= a
			b = scipy.special.gamma(1.0/gamma_range)
			c = scipy.special.gamma(3.0/gamma_range)
			prec_gammas = a/(b*c)

			# solve alpha by guessing values that minimize ro
			pos = np.argmin((prec_gammas - rhat_norm)**2)
			alpha = gamma_range[pos]

			gam1 = scipy.special.gamma(1.0/alpha)
			gam2 = scipy.special.gamma(2.0/alpha)
			gam3 = scipy.special.gamma(3.0/alpha)

			aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
			bl = aggdratio * left_mean_sqrt
			br = aggdratio * right_mean_sqrt

			# Mean parameter
			N = (br - bl)*(gam2 / gam1)  #*aggdratio
			return alpha, N, left_mean_sqrt, right_mean_sqrt

	def compute_aggd(self, values):
		aggd = []
		for value in values:
			pps1, pps2, pps3, pps4 = self.paired_product(value)
			alpha1, N1, lsq1, rsq1 = self.get_aggd_features(pps1)
			alpha2, N2, lsq2, rsq2 = self.get_aggd_features(pps2)
			alpha3, N3, lsq3, rsq3 = self.get_aggd_features(pps3)
			alpha4, N4, lsq4, rsq4 = self.get_aggd_features(pps4)
			aggd.append(alpha1)
			aggd.append(N1)
			aggd.append(lsq1**2)
			aggd.append(rsq1**2)
			aggd.append(alpha2)
			aggd.append(N2)
			aggd.append(lsq2**2)
			aggd.append(rsq2**2)
			aggd.append(alpha3)
			aggd.append(N3)
			aggd.append(lsq3**2)
			aggd.append(rsq3**2)
			aggd.append(alpha4)
			aggd.append( N4)
			aggd.append(lsq4**2)
			aggd.append(rsq4**2)
		return aggd

	def extract_features(self, image):
		features = []
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		gaussian_pyramid = create_gaussian_pyramid(image, 3)
		laplacian_pyramid = create_laplacian_pyramid(gaussian_pyramid)
		max_chromaticity = self.get_max_chromaticity(laplacian_pyramid, 3)
		mscn = self.compute_mscn_transform(max_chromaticity)
		ggd = self.compute_ggd(mscn)
		aggd = self.compute_aggd(mscn)
		features.extend(ggd)
		features.extend(aggd)
		return features

class TextureAndStructuralFeatures():
	def get_lbp(self, image, levels: int):
		radius = 1
		n_points = 8 * radius
		METHOD = 'uniform'
		lbp = []
		for level in range(levels):
			lbp.append(local_binary_pattern(image[level], n_points, radius, METHOD))
		return lbp
	
	def get_pixel_global_lbp(self, img, center, x, y):
		if x >= 0 and y >= 0:
				if img[x][y] >= center:
						new_value = 1
				else:
						new_value = 0
		else:
				new_value = 0
		return new_value

	def threshold(self, value):
		if value >= 0:
				return 1
		return 0

	def get_neighbors(self, image, i, j):
		rows, cols = image.shape

		neighbors = []

		offsets = [(-1, -1), (-1, 0), (-1, 1),(0, 1), (1, 1), (1, 0),(1, -1),(0, -1)]

		for offset in offsets:
				ni, nj = i + offset[0], j + offset[1]
				if 0 <= ni < rows and 0 <= nj < cols:
						neighbors.append(image[ni, nj])
				else:
						neighbors.append(0)

		return neighbors
	
	def get_uniformity(self, img, x, y, neighbor):
		height, width = img.shape
		pixels = []

		center = img[x][y]

		# Calculate initial threshold difference
		thre = abs(self.get_pixel_global_lbp(img, center, x, y-1) - self.get_pixel_global_lbp(img, center, x-1, y-1))
		value = thre
		

		sum_value = 0

		for i in range(len(neighbor)):
				thre1 = self.threshold(neighbor[i] - center)
				# Use current neighbor value if i-1 is out of bounds
				if i == 0:
						thre2 = thre1
				else:
						thre2 = self.threshold(neighbor[i-1] - center)
				sum_value += abs(thre1 - thre2)

		G_sum = 0
		uglbp_pr = value + sum_value

		if ((uglbp_pr) <= 2):
				for i in range(0, 8):
						thre3 = self.threshold(neighbor[i] - center)
						G_sum += abs(thre3)

				gLBP = G_sum
		else:
				gLBP = 9

		return gLBP

	def get_global_lbp(self, image, levels: int):
		lbp_imgs = []
		for level in range(levels):
			height, width= image[level].shape
			img_lbp = np.zeros((height, width),np.uint8)
			for i in range(0, height-1): # Start from 1 and end at height - 2
				for j in range(0, width-1): # Start from 1 and end at width - 2
						neighbours = self.get_neighbors(image[level], i, j)
						img_lbp[i, j] = self.get_uniformity(image[level], i, j, neighbours) # Pass the correct image level
			lbp_imgs.append(img_lbp)
		return lbp_imgs

	def get_lbp_histogram(self, lap, lbp, Bin):
		height, width= lap.shape
		tau = np.zeros((height,width), np.uint8)
		U_lbp = np.zeros((height,width), np.uint8)
		tau = lap
		U_lbp = lbp
		for k in range(10):
				for i in range(height):
						for j in range(width):
								if U_lbp[i,j] == k:
										Bin[k] += tau[i,j]
		return Bin

	def get_weighted_lbp(self, gray_lap, lbp, Bin):
		lbp_weight_bin = []
		for image, lbp_img in zip(gray_lap, lbp):
				Bins = self.get_lbp_histogram(image, lbp_img, Bin)# Pass the correct image level
				lbp_weight_bin.extend(Bins)
		return lbp_weight_bin

	def apply_gamma_transformation(self, image, gamma, levels):
		transformed_imgs = []
		for level in range(levels):
			# Ensure image is in float32 for accurate power computation
			float_img = np.float32(image[level])

			# Apply the transformation I(x, y) = |I(x, y)|^gamma
			transformed_img = np.power(np.abs(float_img), gamma)

			# Normalize to 0-255 and convert back to uint8
			transformed_img = cv2.normalize(transformed_img, None, 0, 255, cv2.NORM_MINMAX)
			transformed_img = np.uint8(transformed_img)
			transformed_imgs.append(transformed_img)

		return transformed_imgs # Return the list of transformed images

	def compute_hog_features(self, gamma_img):
		orient = 9
		pix_per_cell = 8
		cell_per_block = 2

		hog_features, hog_image = hog(gamma_img, 
															orientations=orient,
															pixels_per_cell=(pix_per_cell, pix_per_cell), 
															cells_per_block=(cell_per_block, cell_per_block), 
															visualize=True, 
															feature_vector=False,
															block_norm="L2-Hys")
		# Reshape the hog_features to combine the cells and orientation into a single dimension
		n_blocks_row, n_blocks_col, n_cells_per_block_row, n_cells_per_block_col, n_orientations = hog_features.shape

		# Flatten the last three dimensions (2, 2, 9) into a single dimension (36)
		hog_features = hog_features.reshape((n_blocks_row, n_blocks_col, -1))

		# Now let's iterate over each block to access the block-wise features
		block_features = []

		for i in range(n_blocks_row):
				for j in range(n_blocks_col):
						# Extract the HOG features for the current block
						current_block_features = hog_features[i, j, :]
						block_features.append(current_block_features)

		

		return block_features
	
	def split_mean(self, block_features):
		b = []
		for i in range(len(block_features[0])):
				column = []
				for x in range(len(block_features)):
						column.append(block_features[x][i])

				b.append(column)
		mean = []
		for my_list in b:
				mean_value = sum(my_list) / len(my_list)
				mean.append(mean_value)

				#mean_set.extend(mean)

		return mean

	def get_hog_features(self, images, levels):
		mean_set = []
		for level in range(levels):
				image = images[level]
				hog_feature = self.compute_hog_features(image)
				mean = self.split_mean(hog_feature)
				
				mean_set.extend(mean)

		return mean_set
	
	def extract_features(self, image):
		features = []
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		gaussian_pyramid = create_gaussian_pyramid(image, 3)
		laplacian_pyramid = create_laplacian_pyramid(gaussian_pyramid)
		lbp = self.get_lbp(laplacian_pyramid, 3)
		bin = [0]*10
		uniform_global_lbp = self.get_global_lbp(lbp, 3)
		weighted_lbp = self.get_weighted_lbp(laplacian_pyramid, uniform_global_lbp, bin)
		gamma_img = self.apply_gamma_transformation(laplacian_pyramid, 0.5, 3)
		hog_features = self.get_hog_features(gamma_img, 3)

		features.extend(weighted_lbp)
		features.extend(hog_features)

		return features

class StatisticalDistributionFeatures():
		def rgb_to_xyz(self, rgb):
			"""
			Convert RGB to XYZ using specific transformation equations.

			Parameters:
			rgb (tuple): RGB values as a tuple (R, G, B) with each value in the range [0, 1].

			Returns:
			tuple: A tuple containing the X, Y, Z values.
			"""
			R, G, B = rgb

			# Apply the transformation equations
			X = R * 2.768892 + G * 1.751748 + B * 1.130160
			Y = R * 1.000000 + G * 4.590700 + B * 0.060100
			Z = R * 0.000000 + G * 0.056508 + B * 5.594292

			return X, Y, Z

		def xyz_to_lab(self, X, Y, Z):
				"""
				Convert XYZ to CIELab using the provided formulas.

				Parameters:
				X, Y, Z (float): XYZ values.

				Returns:
				tuple: A tuple containing the L, a, b values.
				"""
				# Reference white values
				X_ref = 94.92
				Y_ref = 100.0
				Z_ref = 107.38

				# Normalize XYZ values
				X /= X_ref
				Y /= Y_ref
				Z /= Z_ref

				# Apply the cube root function with a threshold for linear part
				def f(t):
						delta = 6 / 29
						if t > delta**3:
								return t**(1/3)
						else:
								return (t / (3 * delta**2)) + (4 / 29)

				L = 116 * f(Y) - 16
				a = 500 * (f(X) - f(Y))
				b = 200 * (f(Y) - f(Z))

				return [L, a, b]

		def image_rgb_to_lab(self, img):
			"""
			Convert an entire RGB image to CIELab color space using OpenCV.

			Parameters:
			input_image_path (str): Path to the input RGB image file.

			Returns:
			np.ndarray: A NumPy array containing the CIELab values for each pixel.
			"""
			# Read the image using OpenCV
			
			
			# Check if the image was loaded successfully
			if img is None:
					raise ValueError("Image not found or unable to read")

			# Convert the image to float32 and normalize to [0, 1]
			rgb_array = img.astype('float32') / 255.0

			# Initialize an array for the Lab values
			lab_array = np.zeros((rgb_array.shape[0], rgb_array.shape[1], 3), dtype='float32')

			# Convert each pixel from RGB to Lab
			for i in range(rgb_array.shape[0]):
					for j in range(rgb_array.shape[1]):
							R, G, B = rgb_array[i, j]
							X, Y, Z = self.rgb_to_xyz((R, G, B))
							L, a, b = self.xyz_to_lab(X, Y, Z)
							lab_array[i, j] = (L, a, b)

			return lab_array

		def svd(self, lab_image): 
				param = []

				L, a, b = lab_image[:,:,0], lab_image[:,:,1], lab_image[:,:,2]

				for ch in [L, a, b]:
						U, S, Vt = np.linalg.svd(ch, full_matrices=False)
						
						shape, loc, scale = weibull_min.fit(S, floc=0)
						
						# Append the shape and scale parameters
						param.append(shape)
						param.append(scale)

				return param

		def get_svd(self, image):
			lab_values = self.image_rgb_to_lab(image)
			svd_value = self.svd(lab_values)
			return svd_value

		def extract_features(self, image):
			features = []
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			svd = self.get_svd(image)
			features.extend(svd)
			return features

class MSAEQA():
	def preprocess_image(self, image):
		if isinstance(image, torch.Tensor):
				image = image.cpu().numpy()
		
		if image.ndim == 4:
				image = image[0] 
	
		if image.ndim == 3 and image.shape[0] == 3:
			image = np.transpose(image, (1, 2, 0))
		return image

	def normalization(self, values):
		min_val = min(values)
		max_val = max(values)
		if max_val == min_val:
				return [0.5] * len(values)
		return [(x - min_val) / (max_val - min_val) for x in values]

	def extract_features(self, image):
		image = self.preprocess_image(image)
		statistical_features = StatisticalFeatures()
		texture_and_structural_features = TextureAndStructuralFeatures()
		statistical_distribution_features = StatisticalDistributionFeatures()
		statistical_features = statistical_features.extract_features(image)
		texture_and_structural_features = texture_and_structural_features.extract_features(image)
		statistical_distribution_features = statistical_distribution_features.extract_features(image)
		features = []
		features.extend(statistical_features)
		features.extend(texture_and_structural_features)
		features.extend(statistical_distribution_features)
		normalized_features = self.normalization(features)
		return normalized_features

