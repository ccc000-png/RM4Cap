import json
import logging
import math
import pickle
import random

import torch
from torch import nn

from models.decoder.Caption_bert import CaptionHead

from models.encoders.RFFR import RFFR, RFFR_base
from models.encoders.AugEncoderLayer import AugEncoder
from models.encoders.CMSA import CMSA
from models.encoders.transformer import Transformer

logger = logging.getLogger(__name__)
def build_model(config, pretrained):

    model = RM4Cap(config, pretrained)
    logger.info(f'Model total parameters: {sum(p.numel() for p in model.parameters()):,}')
        # if config.train.save_checkpoints_path:
        #     model.load_state_dict(torch.load(config.train.save_checkpoints_path))
    return model

class RM4Cap(nn.Module):
    def __init__(self, cfgs,pretrained_model):
        super().__init__()
        self.CMSA = cfgs.CMSA
        self.RFFR = cfgs.RFFR
        self.cfgs = cfgs
        if cfgs.CMSA:
            Multi_Encoder = CMSA(d_model=cfgs.encoder.hidden_dim,max_align= cfgs.encoder.align_objects,)
        elif cfgs.Base==1:
            Multi_Encoder = Transformer(d_model=cfgs.encoder.hidden_dim,max_align= cfgs.encoder.align_objects,)
        if cfgs.RFFR:
            Fine_Grain_Encoder = RFFR(d_model=cfgs.encoder.hidden_dim)
        else:
            Fine_Grain_Encoder = RFFR_base(d_model=cfgs.encoder.hidden_dim)

        self.AugInformation = AugEncoder(MultiEncoder= Multi_Encoder,
                                         Fine_Grain_Encoder=Fine_Grain_Encoder,
                                           visual_dim = cfgs.encoder.visual_dim,
                                           object_dim = cfgs.encoder.object_dim,
                                           hidden_dim = cfgs.encoder.hidden_dim)
        '''2.字幕头'''
        # self.visual_encoder = pretrained_model.visual
        state_dict = pretrained_model.state_dict()
        embed_dim = state_dict["text_projection"].shape[1]
        self.vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        pretrained_embedding = {k.lstrip("token_embedding."): v for k, v in state_dict.items()
                                if k.startswith("token_embedding")}
        logger.info(f'Model total parameters: {sum(p.numel() for p in self.AugInformation.parameters()):,}')
        self.caption_head = CaptionHead(hidden_dim = cfgs.encoder.hidden_dim,
                                        word_embedding_size=transformer_width,
                                        visual_feature_size=embed_dim,
                                        pretrained_embedding=pretrained_embedding,
                                        max_v_len=cfgs.sample_numb + 1,
                                        max_t_len=cfgs.decoder.max_caption_len,  # 77
                                        hidden_size=embed_dim,
                                        vocab_size=self.vocab_size,
                                        fusion_sem=cfgs.fusion_sem,
                                        fusion_fine=cfgs.fusion_fine)

    def forward(self, global_visual, input_ids, input_mask,pair):
        aug_object_features, aug_action_features= self.AugInformation(
            visual = global_visual,
            pair=pair,
            objects = None,
        )

        prediction_scores = self.caption_head(
            global_visual,
            aug_object_features,
            aug_action_features,
            input_ids,
            input_mask,
        )
        return {"prediction_scores":prediction_scores}
