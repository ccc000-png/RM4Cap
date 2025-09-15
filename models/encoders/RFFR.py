import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from torch import nn

class RFFR_base(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()

        self._reset_parameters()
        #
        self.d_model = d_model

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):

        tgt_norm = tgt / tgt.norm(dim=-1, keepdim=True)  # (128, 15, 4, 512)

        cosine_sim_matrix = torch.matmul(tgt_norm, tgt_norm.transpose(1, 2))  # (64, 8, 10)



        inverse_similarity_matrix = cosine_sim_matrix
        inverse_similarity_matrix = torch.tanh(inverse_similarity_matrix)

        weighted_tgt = torch.bmm(inverse_similarity_matrix, tgt)
        output = weighted_tgt

        return output

class RFFR(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()

        self._reset_parameters()
        #
        self.d_model = d_model
        self.w = nn.Linear(d_model, 1)
        self.Uo = nn.Linear(d_model, d_model)
        self.bo = nn.Parameter(torch.ones(d_model), requires_grad=True)
        self.wo = nn.Linear(d_model, 1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, tgt):
        """ RFFR """
        """
        src:visual
        tgt:targets
        """
        # U_objs = self.Uo(tgt)
        # attn_feat = U_objs + self.bo  # (bsz, sample_numb, max_objects, hidden_dim)
        # attn_weights = self.wo(torch.tanh(attn_feat))  # (bsz, sample_numb, max_objects, 1)
        # attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
        # attn_objects = attn_weights * attn_feat
        # attn_objects = attn_objects.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)


        frame=tgt.shape[1]
        new_tgt = []
        for i in range(frame):
            A_tgt = tgt[:, i, :, :]
            weights=0.0
            for j in range(frame):
                if i==j:
                    continue;
                B_tgt = tgt[:, j, :, :]
                weight = cosine_similarity(A_tgt, B_tgt)

                weights+=weight.softmax(dim=-2)
            output = torch.bmm(weights, A_tgt)
            new_tgt.append(output.sum(dim=-2))

        return torch.stack(new_tgt,dim=1)

def cosine_similarity(vec1, vec2, flag=None):
    vec1_norm = vec1 / vec1.norm(dim=-1, keepdim=True)  # (64, 8, 512)
    vec2_norm = vec2 / vec2.norm(dim=-1, keepdim=True)  # (64, 8, 512)
    cosine_sim_matrix = torch.bmm(vec1_norm, vec2_norm.transpose(1, 2))  # (64, 8, 10)


    sim_min = cosine_sim_matrix.view(cosine_sim_matrix.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1)
    sim_max = cosine_sim_matrix.view(cosine_sim_matrix.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1)

    normalized_cosine_sim_matrix = (cosine_sim_matrix - sim_min) / (sim_max - sim_min)
    return 1-normalized_cosine_sim_matrix

