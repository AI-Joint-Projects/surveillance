import torch 
import torch.nn as nn
import torch.nn.functional as F

class CosFaceLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, scale=30.0, margin=0.35):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, features, labels):
        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)
        
        # Calculate cosine similarities
        cos_theta = F.linear(features, weights)  # shape: [N, num_classes]
        
        # Get cosine similarity for ground truth classes
        batch_size = features.size(0)
        ground_truth_cos = cos_theta[torch.arange(batch_size), labels].view(-1, 1)  # shape: [N, 1]
        
        # Apply margin to ground truth class
        ground_truth_cos_m = ground_truth_cos - self.margin
        
        # Calculate exp(s*cos(theta)) for all classes
        exp_logits = torch.exp(self.scale * cos_theta)  # shape: [N, num_classes]
        
        # Calculate exp(s*(cos(theta_yi) - m)) for ground truth classes
        exp_margins = torch.exp(self.scale * ground_truth_cos_m)  # shape: [N, 1]
        
        # Sum exp(s*cos(theta_j)) for j ≠ yi
        sum_exp_logits = exp_logits.sum(dim=1) - exp_logits[torch.arange(batch_size), labels]
        
        # Final loss following the paper formula exactly
        loss = -torch.log(exp_margins.view(-1) / (exp_margins.view(-1) + sum_exp_logits)).mean()
        
        return loss
    
