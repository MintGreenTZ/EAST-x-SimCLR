import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import EAST
import os
from dataset import get_rotate_mat
import numpy as np
import lanms
from gaussian_blur import GaussianBlur
import time

def resize_img(img):
	'''resize image to be divisible by 32
	'''
	w, h = img.size
	resize_w = w
	resize_h = h

	resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
	resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
	img = img.resize((resize_w, resize_h), Image.BILINEAR)
	ratio_h = resize_h / h
	ratio_w = resize_w / w

	return img, ratio_h, ratio_w


def load_pil(img):
	'''convert PIL Image to torch.Tensor
	'''
	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
	return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
	'''check if the poly in image scope
	Input:
		res        : restored poly in original image
		score_shape: score map shape
		scale      : feature map -> image
	Output:
		True if valid
	'''
	cnt = 0
	for i in range(res.shape[1]):
		if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
			cnt += 1
	return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
	'''restore polys from feature maps in given positions
	Input:
		valid_pos  : potential text positions <numpy.ndarray, (n,2)>
		valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
		score_shape: shape of score map
		scale      : image / feature map
	Output:
		restored polys <numpy.ndarray, (n,8)>, index
	'''
	polys = []
	index = []
	valid_pos *= scale
	d = valid_geo[:4, :] # 4 x N
	angle = valid_geo[4, :] # N,

	for i in range(valid_pos.shape[0]):
		x = valid_pos[i, 0]
		y = valid_pos[i, 1]
		y_min = y - d[0, i]
		y_max = y + d[1, i]
		x_min = x - d[2, i]
		x_max = x + d[3, i]
		rotate_mat = get_rotate_mat(-angle[i])
		
		temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
		temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
		coordidates = np.concatenate((temp_x, temp_y), axis=0)
		res = np.dot(rotate_mat, coordidates)
		res[0,:] += x
		res[1,:] += y
		
		if is_valid_poly(res, score_shape, scale):
			index.append(i)
			polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
	return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
	'''get boxes from feature map
	Input:
		score       : score map from model <numpy.ndarray, (1,row,col)>
		geo         : geo map from model <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms
	Output:
		boxes       : final polys <numpy.ndarray, (n,9)>
	'''
	score = score[0,:,:]
	xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	if xy_text.size == 0:
		return None

	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
	polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape) 
	if polys_restored.size == 0:
		return None

	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
	boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
	return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
	'''refine boxes
	Input:
		boxes  : detected polys <numpy.ndarray, (n,9)>
		ratio_w: ratio of width
		ratio_h: ratio of height
	Output:
		refined boxes
	'''
	if boxes is None or boxes.size == 0:
		return None
	boxes[:,[0,2,4,6]] /= ratio_w
	boxes[:,[1,3,5,7]] /= ratio_h
	return np.around(boxes)
	
	
def detect(img, model, device):
	'''detect text regions of img using model
	Input:
		img   : PIL Image
		model : detection model
		device: gpu if gpu is available
	Output:
		detected polys <numpy.ndarray, (n,9)>
	'''
	img, ratio_h, ratio_w = resize_img(img)
	with torch.no_grad():
		score, geo, merged_feature = model(load_pil(img).to(device))

	score_cp = score.cpu().numpy()[0, 0, ...]
	# print(score.flatten().shape)
	j = Image.fromarray(((score_cp - np.min(score_cp.flatten())) / (
				np.max(score_cp.flatten()) - np.min(score_cp.flatten()) + 1e-8) * 255).astype(np.uint8))

	boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
	return adjust_ratio(boxes, ratio_w, ratio_h), j	#,score  #打印score map


def plot_boxes(img, boxes):
	'''plot boxes on image
	'''
	if boxes is None:
		return img
	
	draw = ImageDraw.Draw(img)
	for box in boxes:
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
		draw.polygon([box[0]+1, box[1]+1, box[2]+1, box[3]+1, box[4]+1, box[5]+1, box[6]+1, box[7]+1], outline=(0, 255, 0))
	return img




def detect_dataset(model, device, test_img_path, submit_path):
	'''detection on whole dataset, save .txt results in submit_path
	Input:
		model        : detection model
		device       : gpu if gpu is available
		test_img_path: dataset path
		submit_path  : submit result for evaluation
	'''
	img_files = os.listdir(test_img_path)
	img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])

	avg_time = 0

	for i, img_file in enumerate(img_files):
		# print('evaluating {} image'.format(i), end='\r')
		start_time = time.time()
		boxes, j = detect(Image.open(img_file), model, device)
		cost_time = time.time() - start_time
		avg_time += cost_time
		seq = []
		if boxes is not None:
			seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes])
		with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg','.txt')), 'w') as f:
			f.writelines(seq)
		print('evaluating {} image'.format(i), end='\r')

		# break
	avg_time = avg_time/len(img_files)
	print(avg_time)

if __name__ == '__main__':
	img_path    = '../ICDAR_2015/test_img/img_15.jpg'
	# model_path = '/home/weiran/EAST/pths/model_epoch_600.pth'
	model_path  = '/home/weiran/ICDAR_2015/simclr15_pths/model_epoch_750_exp_11.pth'
	res1_img     = './res1.bmp'
	res2_img 	= './res2.bmp'
	img_ori = './img_15.jpg'
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(False).to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()

	img = Image.open(img_path)
	img = img.convert("RGB")
	length = 512
	data_transforms = transforms.Compose([GaussianBlur(kernel_size=int(0.1 * length))])
	adjust_transform = transforms.Compose([transforms.ColorJitter(0.8, 0.8, 0.8, 0.25)])
	img1 = adjust_transform(img)
	# print(type(img1))
	img2 = data_transforms(img1)
	img2 = Image.fromarray(img2)
	# print(type(img2))
	# img.save('img.jpg')
	# img1.save('img1.jpg')
	# img2.save('img2.jpg')

	boxes, j = detect(img, model, device)
	# j.save('pred_score1.jpg')
	plot_img = plot_boxes(img, boxes)
	plot_img.save(img_ori)

	# boxes, j = detect(img1, model, device)
	# j.save('pred_score1.jpg')
	# plot_img = plot_boxes(img1, boxes)
	# plot_img.save(res1_img)
	#
	# boxes, j = detect(img2, model, device)
	# j.save('pred_score2.jpg')
	# plot_img = plot_boxes(img2, boxes)
	# plot_img.save(res2_img)
