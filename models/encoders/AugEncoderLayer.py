import torch
from torch import nn

class AugEncoder(nn.Module):
    def __init__(self, MultiEncoder, Fine_Grain_Encoder, visual_dim, object_dim,
                                           hidden_dim):
        super(AugEncoder, self).__init__()
        self.video_embeddings = nn.Linear(visual_dim, hidden_dim)
        if object_dim is not None:
            self.object_embeddings = nn.Linear(object_dim, hidden_dim)
        self.cmsa_module = MultiEncoder
        self.rffr_module = Fine_Grain_Encoder

    def forward(self, visual, pair, objects = None):
        vhidden_states = self.video_embeddings(visual)

        align_hidden_states= self.cmsa_module(vhidden_states, pair,mask=None)

        fine_grain_features = self.rffr_module(align_hidden_states)

        return align_hidden_states, fine_grain_features