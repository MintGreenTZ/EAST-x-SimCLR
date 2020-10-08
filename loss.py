import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from simclr_projector import SimCLRProjector


def get_ce_loss(gt_score, pred_score):
	gt_score = gt_score.squeeze(1).long()
	b, h, w = gt_score.shape
	beta = 1 - torch.sum(gt_score) / (b * h * w)
	weight = torch.ones(2)
	weight = weight.cuda()
	weight[1] = 1 - beta
	weight[0] = beta

	pred_score = torch.cat((1 - pred_score, pred_score), dim=1)
	ce_loss = F.cross_entropy(pred_score, gt_score, weight=weight)  # , weight=weight
	return ce_loss


def get_dice_loss(gt_score, pred_score):
	inter = torch.sum(gt_score * pred_score)
	union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
	return 1 - (2 * inter / union)


def get_geo_loss(gt_geo, pred_geo):
	d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
	d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
	area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
	area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
	w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
	h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
	area_intersect = w_union * h_union
	area_union = area_gt + area_pred - area_intersect
	iou_loss_map = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
	angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
	return iou_loss_map, angle_loss_map


class Loss(nn.Module):
	def __init__(self, device, batch_size, temperature, use_cosine_similarity, weight_angle=10):
		super(Loss, self).__init__()
		# east
		self.weight_angle = weight_angle
		# simclr
		self.batch_size = batch_size
		self.temperature = temperature
		self.device = device
		self.softmax = torch.nn.Softmax(dim=-1)
		self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
		self.similarity_function = self._get_similarity_function(use_cosine_similarity)
		self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
		self.model = SimCLRProjector().to(self.device)

	def _get_similarity_function(self, use_cosine_similarity):
		if use_cosine_similarity:
			self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
			return self._cosine_simililarity
		else:
			return self._dot_simililarity

	def _get_correlated_mask(self):
		diag = np.eye(2 * self.batch_size)
		l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
		l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
		mask = torch.from_numpy((diag + l1 + l2))
		mask = (1 - mask).type(torch.bool)
		return mask.to(self.device)

	@staticmethod
	def _dot_simililarity(x, y):
		v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
		# x shape: (N, 1, C)
		# y shape: (1, C, 2N)
		# v shape: (N, 2N)
		return v

	def _cosine_simililarity(self, x, y):
		# x shape: (N, 1, C)
		# y shape: (1, 2N, C)
		# v shape: (N, 2N)
		v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
		return v

	def traditional_east_loss(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
		if torch.sum(gt_score) < 1:
			return torch.sum(pred_score + pred_geo) * 0

		# classify_loss = 0.7*(get_dice_loss(gt_score, pred_score*(1-ignored_map))) + 0.3*(get_ce_loss(gt_score, pred_score*(1-ignored_map)))
		classify_loss = get_dice_loss(gt_score, pred_score * (1 - ignored_map))
		iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)

		angle_loss = torch.sum(angle_loss_map * gt_score) / torch.sum(gt_score)
		iou_loss = torch.sum(iou_loss_map * gt_score) / torch.sum(gt_score)
		geo_loss = self.weight_angle * angle_loss + iou_loss
		print('classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss, angle_loss,
																						 iou_loss))
		return geo_loss + classify_loss

	def nt_xent_loss(self, zis, zjs):
		representations = torch.cat([zjs, zis], dim=0)
		# print(representations[0,...])
		similarity_matrix = self.similarity_function(representations, representations)
		# print((similarity_matrix.data.cpu().numpy()*100).astype(np.int))
		# filter out the scores from the positive samples
		l_pos = torch.diag(similarity_matrix, self.batch_size)
		r_pos = torch.diag(similarity_matrix, -self.batch_size)
		positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

		negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

		logits = torch.cat((positives, negatives), dim=1)
		logits /= self.temperature

		labels = torch.zeros(2 * self.batch_size).to(self.device).long()
		loss = self.criterion(logits, labels)

		return loss / (2 * self.batch_size)

	def simclr_loss(self, xis, xjs):

		# print("size of xis")
		# print(xis.size())

		zis = self.model(xis)  # [N,C]
		zjs = self.model(xjs)  # [N,C]

		# normalize projection feature vectors
		zis = F.normalize(zis, dim=1)
		zjs = F.normalize(zjs, dim=1)

		loss = self.nt_xent_loss(zis, zjs)
		return loss

	def forward(self, gt_score1, pred_score1, gt_geo1, pred_geo1, ignored_map1, merged_feature1,
				gt_score2, pred_score2, gt_geo2, pred_geo2, ignored_map2, merged_feature2, epoch):
		east_loss = self.traditional_east_loss(gt_score1, pred_score1, gt_geo1, pred_geo1, ignored_map1) \
					+ self.traditional_east_loss(gt_score2, pred_score2, gt_geo2, pred_geo2, ignored_map2)
		simclr_loss = self.simclr_loss(merged_feature1, merged_feature2)

		print('east loss is {:.8f}, simclr loss is {:.8f}'.format(east_loss, simclr_loss))
		# return east_loss + simclr_loss

		if epoch <100 :
			return east_loss
		else :
			return east_loss + simclr_loss #权重不加这里