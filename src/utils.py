import numpy as np

from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

def compare_psnr_ssim(img_i, img_t):
	avg_psnr = 0
	avg_ssim = 0
	for i, t in zip(img_i, img_t):
		i = np.array(UnNormalize(i)*255, dtype=np.uint8).transpose(1, 2, 0)[:, :, ::-1]
		t = np.array(UnNormalize(t)*255, dtype=np.uint8).transpose(1, 2, 0)[:, :, ::-1]

		avg_psnr += PSNR(i, t, data_range=255) / img_i.shape[0]
		avg_ssim += SSIM(i, t, data_range=255, multichannel=True) / img_i.shape[0]

	return avg_psnr, avg_ssim

def UnNormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
	img = tensor.cpu().detach().clone()
	for c, m, s in zip(img, mean, std):
		c.mul_(s).add_(m)
	return img