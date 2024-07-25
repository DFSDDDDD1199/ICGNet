import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch_scatter as ts
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, target, input2):
        target1 = torch.squeeze(target, dim=1)
        if input2.dim()>2:
            input2 = input2.view(input2.size(0), input2.size(1),-1)  # N,C,H,W => N,C,H*W
            input2 = input2.transpose(1,2)    # N,C,H*W => N,H*W,C
            input2 = input2.contiguous().view(-1,input2.size(2))   # N,H*W,C => N*H*W,C
        target2 = target1.view(-1,1).long()

        logpt = F.log_softmax(input2 + 1e-4, dim=1)
        # print(logpt.size())
        # print(target2.size())
        logpt = logpt.gather(1,target2)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input2.data.type():
                self.alpha = self.alpha.type_as(input2.data)
            at = self.alpha.gather(0,target.data.view(-1).type(torch.int64))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



class KeypointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=0.9, gamma=2)
    
    def forward(self, keypoint_est, keypoint_label):
        return self.focal_loss(keypoint_label, keypoint_est)


class DistillLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='mean', log_target=True)

    def forward(self, input_scores, target_scores):
        b, m, n = input_scores.shape
        input_scores = torch.clamp(input_scores, min=-100, max=0.0)
        target_scores = torch.clamp(target_scores, min=-100, max=0.0)

        assert input_scores.size()[-1] == 1025
        assert input_scores.size()[-2] == 1025
        assert target_scores.size()[-1] == 1025
        assert target_scores.size()[-2] == 1025

        valid_row = torch.ones_like(input_scores).bool()
        valid_row[:, -1, :] = False
        valid_col = torch.ones_like(input_scores).bool()
        valid_col[:, :, -1] = False

        input_row_sum = input_scores.exp().sum(dim=-1, keepdim=True).log()
        input_col_sum = input_scores.exp().sum(dim=-2, keepdim=True).log()
        target_row_sum = target_scores.exp().sum(dim=-1, keepdim=True).log()
        target_col_sum = target_scores.exp().sum(dim=-2, keepdim=True).log()

        input_row_prob = (input_scores - input_row_sum)[valid_row]
        input_col_prob = (input_scores - input_col_sum)[valid_col]
        target_row_prob = (target_scores - target_row_sum)[valid_row]
        target_col_prob = (target_scores - target_col_sum)[valid_col]   
        
        return 0.5 * m * self.criterion(input=input_row_prob, target=target_row_prob) + 0.5 * n * self.criterion(input=input_col_prob, target=target_col_prob)

        
class MatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, scores, keypoints0, keypoints1, disp_gt):
        assert scores.size()[-1] == keypoints0.size()[1] + 1

        def torch_find_matches_disparity(src_keypoints1, src_keypoints2, disparity, dist_thresh=2, n_iters=2):
            match_list_1, match_list_2 = torch.empty(0,dtype=torch.int64,device=src_keypoints1.device), torch.empty(0,dtype=torch.int64,device=src_keypoints2.device) 
            missing_indices_1 = torch.arange(len(src_keypoints1), device=src_keypoints1.device, dtype=torch.long)
            missing_indices_2 = torch.arange(len(src_keypoints2), device=src_keypoints2.device, dtype=torch.long)
            src_keypoints1_projected = warp_keypoints_disparity(src_keypoints1, disparity) #keypoint1 must be projected to keypoint2 space for comparsion
            for i in range(n_iters): #2nd iteration only recover 1 or 2 keypoint matches in most cases. So 1 should be enough.
                keypoints1 = src_keypoints1_projected[missing_indices_1, :]
                keypoints2 = src_keypoints2[missing_indices_2, :]
                distance = torch_cdist(keypoints1, keypoints2)
                min1 = torch.argmin(distance, 1)
                min2 = torch.argmin(distance, 0)
                intersect_indexes_2 = torch.where(min1[min2] == torch.arange(len(min2), device=min1.device))[0]
                intersect_indexes1 = min2[intersect_indexes_2]
                matched_distances = distance[intersect_indexes1, intersect_indexes_2] < dist_thresh
                intersect_indexes1 = intersect_indexes1[matched_distances]
                intersect_indexes_2 = intersect_indexes_2[matched_distances]
                matched_indexes_1 = missing_indices_1[intersect_indexes1]
                matched_indexes_2 = missing_indices_2[intersect_indexes_2]
                missing_indices_1 = torch_setdiff1d(missing_indices_1, matched_indexes_1)
                missing_indices_2 = torch_setdiff1d(missing_indices_2, matched_indexes_2)
                match_list_1 = torch.cat((match_list_1, matched_indexes_1))
                match_list_2 = torch.cat((match_list_2, matched_indexes_2))
            return match_list_1, match_list_2, missing_indices_1, missing_indices_2

        def torch_cdist(keypoints1, keypoints2):
            diff = (keypoints1[:, None, :] - keypoints2[None, :, :]) ** 2
            summed = diff.sum(-1)
            distance = torch.sqrt(summed)
            return distance

        def torch_setdiff1d(miss_index, match_index):
            combined = torch.cat((miss_index, match_index))
            unq, count = combined.unique(return_counts=True)
            diff = unq[count == 1]
            return diff

        def warp_keypoints_disparity(keypoints, disparity):
            '''
            keypoints.shape = (n, 2)
            disparity.shape = (h, w)
            '''
            keypoints_disparity = torch.tensor([disparity[p[1].int()][p[0].int()] for p in keypoints]).to(keypoints.device)
            warped_keypoints = keypoints.clone()
            warped_keypoints[:, 0] -= keypoints_disparity
            return warped_keypoints

        all_match_index_0, all_match_index_1, all_match_index_2 = torch.empty(0,dtype=torch.int64,device=disp_gt.device), torch.empty(0,dtype=torch.int64,device=disp_gt.device), torch.empty(0,dtype=torch.int64,device=disp_gt.device)
        bs = keypoints0.size()[0]
        for k in range(bs):
            ma_0, ma_1, miss_0, miss_1 = torch_find_matches_disparity(keypoints0[k], keypoints1[k], disp_gt[k], dist_thresh=2, n_iters=2)
            all_match_index_0 = torch.cat([all_match_index_0, torch.empty(len(ma_0) + len(miss_0) + len(miss_1), dtype=torch.long, device=ma_0.device).fill_(k)])
            all_match_index_1 = torch.cat([all_match_index_1, ma_0, miss_0, torch.empty(len(miss_1), dtype=torch.long, device=miss_1.device).fill_(-1)])
            all_match_index_2 = torch.cat([all_match_index_2, ma_1, torch.empty(len(miss_0), dtype=torch.long, device=miss_0.device).fill_(-1), miss_1])
        gt_indexes = torch.stack([all_match_index_0, all_match_index_1, all_match_index_2], -1)
        neg_flag = (gt_indexes[:, 1] == -1) | (gt_indexes[:, 2] == -1)
        pos_weight = neg_flag.sum() / (neg_flag.sum() + (~neg_flag).sum())
        neg_weight = (~neg_flag).sum() / (neg_flag.sum() + (~neg_flag).sum())
        loss_pre_components = scores[gt_indexes[:, 0], gt_indexes[:, 1], gt_indexes[:, 2]]
        loss_pre_components = torch.clamp(loss_pre_components, min=-100, max=0.0)
        loss_vector = -1 * loss_pre_components
        neg_index, pos_index = gt_indexes[:, 0][neg_flag], gt_indexes[:, 0][~neg_flag]
        batched_pos_loss, batched_neg_loss = ts.scatter_mean(loss_vector[~neg_flag], pos_index, dim_size=bs), ts.scatter_mean(loss_vector[neg_flag], neg_index, dim_size=bs)
        pos_loss, neg_loss = pos_weight * batched_pos_loss.mean(), neg_weight * batched_neg_loss.mean()
        loss = pos_loss + neg_loss
        return loss




