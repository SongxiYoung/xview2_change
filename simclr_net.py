"""
SimCLR model
T. Chen, et al. “A Simple Framework for Contrastive Learning of Visual Representations,” ICML 2020.
Pre-text task
Both pre- and post-disaster bldg objects, one sides
@ 2021 Bo Peng (bo.peng@wisc.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet

class SimCLR_Net(nn.Module):
    """
    Object based SimCLR Net
    Both bitemporal pre- & post- images are needed
    """

    def __init__(self, h_dim = 512, z_dim = 128, backbone_net='resnet18'):
        super(SimCLR_Net, self).__init__()

        # load pre-trained ResNet as backbone
        # default is ResNet18
        if backbone_net == 'ResNet34':
            self.encode = resnet.ResNet34_Encode(h_dim, in_channels=3)
        elif backbone_net == 'resnet50':
            self.encode = resnet.ResNet50_Encode(h_dim, in_channels=3)
        else:
            self.encode = resnet.ResNet18_Encode(h_dim, in_channels=3)

        self.project = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Linear(h_dim // 2, z_dim)
        )


    def NT_Xent_Loss_v3(self, z_pre, z_post, tau=0.1):
        assert isinstance(z_pre, torch.Tensor)
        assert isinstance(z_post, torch.Tensor)
        assert z_pre.shape == z_post.shape

        device = z_post.device
        loss = torch.tensor(0.0, device=device)

        # similarity between pairs of pre+post representations
        n_samples, n_features = z_post.shape # Batch size, Feature dimensions

        # pairs similarity
        for i in range(n_samples):
            sim_positive = (F.cosine_similarity(z_pre[i], z_post[i], dim=0) / tau).exp() # positive pairs similarity

            curr_pre_batch = torch.cat([z_pre[[i]]] * (n_samples - 1), dim=0)
            curr_post_batch = torch.cat([z_post[[i]]] * (n_samples - 1), dim=0)
            others_pre_batch = torch.cat([z_pre[:i], z_pre[i+1:]], dim=0)
            others_post_batch = torch.cat([z_post[:i], z_post[i + 1:]], dim=0)

            sim_negative = (F.cosine_similarity(curr_pre_batch, others_pre_batch, dim=1) / tau).exp().sum() \
                           + (F.cosine_similarity(curr_pre_batch, others_post_batch, dim=1) / tau).exp().sum() \
                           + (F.cosine_similarity(curr_post_batch, others_pre_batch, dim=1) / tau).exp().sum() \
                           + (F.cosine_similarity(curr_post_batch, others_post_batch, dim=1) / tau).exp().sum() \
                           + sim_positive

            loss = loss - torch.log(sim_positive / sim_negative)

        loss = loss / n_samples

        return loss

    def pairs_similarity(self, z_pre, z_post, metric = 'l2'):
        if metric == 'cosine': # cosine similarity
            return F.cosine_similarity(z_pre, z_post, dim=1)
        else: # l2 norm of difference vector
            z_pre = F.normalize(z_pre, p=2, dim=1)
            z_post = F.normalize(z_post, p=2, dim=1)
            return torch.linalg.norm(z_pre-z_post, dim=1)

    def forward(self, x_pre, x_post):

        # image latent representations
        h_pre = self.encode(x_pre)
        h_post = self.encode(x_post)

        # projections of latent representations
        z_pre = self.project(h_pre)
        z_post = self.project(h_post)

        # l2 normalization of projections
        #z_pre = F.normalize(z_pre, p=2, dim=1)
        #z_post = F.normalize(z_post, p=2, dim=1)

        return h_pre, h_post, z_pre, z_post
