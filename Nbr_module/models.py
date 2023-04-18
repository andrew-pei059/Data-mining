import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Tanh 被限制在 +- 1，
class DimensionReduction(nn.Module):
    def __init__(self, ini_rev_dim=768, emb_dim=512):
        super(DimensionReduction, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(ini_rev_dim, emb_dim),
            # nn.Tanh(),
            # nn.Linear(256, 512),
            # nn.Tanh()
        )

    def forward(self, batch):
        dim_output = self.seq(batch)
        
        return dim_output
    
class ComponentAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(ComponentAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, batch):
        # batch.shape => (batch, seq_len, embed_dim)
        attn_output, attn_output_weights = self.multihead_attn(batch, batch, batch)
        
        return attn_output
    
class Coattention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(Coattention, self).__init__()
        self.multihead_attn_1 = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.multihead_attn_2 = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self, rev_batch, batch_2):
        # batch_2 may be reviewEmb or user/item embedding
        attn_output, attn_output_weights = self.multihead_attn_1(rev_batch, batch_2, batch_2)  # rev_batch shape
        rev_batch_output = torch.mean(attn_output, 1)
        
        attn_output, attn_output_weights = self.multihead_attn_2(batch_2, rev_batch, rev_batch)  # batch_2 shape
        batch_2_output = torch.mean(attn_output, 1)

        return rev_batch_output, batch_2_output
    
class Neighbor_interacted_module(nn.Module):
    def __init__(self, ini_rev_dim, emb_dim, num_heads):
        super(Neighbor_interacted_module, self).__init__()
        # DimensionReduction 將維度從 ini_rev_dim 變成 emb_dim
        self.rev_batch_rd = DimensionReduction(ini_rev_dim=ini_rev_dim, emb_dim=emb_dim)
        self.rev_batch_attn = ComponentAttention(emb_dim=emb_dim, num_heads=num_heads)
        self.batch_2_attn = ComponentAttention(emb_dim=emb_dim, num_heads=num_heads)
        self.co_attn = Coattention(emb_dim=emb_dim, num_heads=num_heads)
        
    def forward(self, rev_batch, batch_2):
        rev_batch = self.rev_batch_rd(rev_batch)
        rev_batch_attn = self.rev_batch_attn(rev_batch)
        batch_2_attn = self.batch_2_attn(batch_2)
        nbr_review_vector, latent_vector = self.co_attn(rev_batch_attn, batch_2_attn)
        
        return nbr_review_vector, latent_vector
    
class User_Item_interacted_module(nn.Module):
    def __init__(self, ini_rev_dim, emb_dim, num_heads):
        super(User_Item_interacted_module, self).__init__()
        self.rd_1 = DimensionReduction(ini_rev_dim=ini_rev_dim, emb_dim=emb_dim)
        self.rd_2 = DimensionReduction(ini_rev_dim=ini_rev_dim, emb_dim=emb_dim)
        self.co_attn = Coattention(emb_dim=emb_dim, num_heads=num_heads)
        
    def forward(self, user_rev_batch, item_rev_batch):
        user_rev_batch = self.rd_1(user_rev_batch)
        item_rev_batch = self.rd_2(item_rev_batch)
        user_rev_vector, item_rev_vector = self.co_attn(user_rev_batch, item_rev_batch)
        
        return user_rev_vector, item_rev_vector
    
class Integrated_Neighbor_module(nn.Module):
    def __init__(self, ini_rev_dim, emb_dim, num_heads):
        super(Integrated_Neighbor_module, self).__init__()
        self.user_nbr = Neighbor_interacted_module(ini_rev_dim=ini_rev_dim, emb_dim=emb_dim, num_heads=num_heads)
        self.review_interacted = User_Item_interacted_module(ini_rev_dim=ini_rev_dim, emb_dim=emb_dim, num_heads=num_heads)
        self.item_nbr = Neighbor_interacted_module(ini_rev_dim=ini_rev_dim, emb_dim=emb_dim, num_heads=num_heads)
        self.attn = ComponentAttention(emb_dim=emb_dim, num_heads=num_heads)  # embed_dim 需更改
        self.seq = nn.Sequential(
            nn.Linear( emb_dim+emb_dim, 512 ), # 目前是串接，ok 嗎 ???
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    def forward(self, user_emb, user_itemEmb, user_nbr_revEmb, user_revEmb, item_revEmb, item_emb, item_nbr_revEmb, item_userEmb):
        # user/item side nbr interaction module 
        user_nbr_rev_vector, user_item_vector = self.user_nbr(user_nbr_revEmb, user_itemEmb)
        user_rev_vector, item_rev_vector = self.review_interacted(user_revEmb, item_revEmb)
        item_nbr_rev_vector, item_user_vector = self.item_nbr(item_nbr_revEmb, item_userEmb)

        # merge user side vectors
        user_nbr_rev_vector, user_item_vector = user_nbr_rev_vector.unsqueeze(1), user_item_vector.unsqueeze(1)
        user_emb, user_rev_vector = user_emb.unsqueeze(1), user_rev_vector.unsqueeze(1)
        user_nbr_interaction_vector = torch.cat( [user_nbr_rev_vector, user_item_vector, user_emb, user_rev_vector], 1 )
        # 用 attention 對 4 個 user 相關的向量做 reweight
        user_nbr_interaction_vector = self.attn(user_nbr_interaction_vector)
        user_nbr_interaction_vector = torch.mean(user_nbr_interaction_vector, 1)

        # merge item side vectors
        item_nbr_rev_vector, item_user_vector = item_nbr_rev_vector.unsqueeze(1), item_user_vector.unsqueeze(1)
        item_emb, item_rev_vector = item_emb.unsqueeze(1), item_rev_vector.unsqueeze(1)
        item_nbr_interaction_vector = torch.cat( [item_nbr_rev_vector, item_user_vector, item_emb, item_rev_vector], 1 )
        # 用 attention 對 4 個 item 相關的向量做 reweight
        item_nbr_interaction_vector = self.attn(item_nbr_interaction_vector)
        item_nbr_interaction_vector = torch.mean(item_nbr_interaction_vector, 1)

        # MLP ----------------------------
        fc_input = torch.cat( [user_nbr_interaction_vector, item_nbr_interaction_vector], 1 )
        fc_output = self.seq(fc_input)

        return fc_output