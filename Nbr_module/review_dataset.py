# %%
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# %%
class ReviewDataset(Dataset):
    def __init__(self, target='other'):
        if target =="train":
            self.review_df = pd.read_pickle('data/train_data.pkl')
        elif target =="test":
            self.review_df = pd.read_pickle('data/test_data.pkl')

        # User
        self.user_id_emb = pd.read_pickle('data/User/user_id_emb.pkl')
        self.user_interacted_appEmb = pd.read_pickle('data/User/user_interacted_appEmb.pkl')
        self.user_id_nbrReviewsEmb = pd.read_pickle('data/User/user_id_nbrReviewsEmb.pkl')
        self.user_id_ReviewsEmb = pd.read_pickle('data/User/user_id_ReviewsEmb.pkl')
        # Item
        self.item_id_emb = pd.read_pickle('data/App/app_id_emb.pkl')
        self.item_interacted_userEmb = pd.read_pickle('data/App/app_interacted_userEmb.pkl')
        self.item_id_nbrReviews_emb = pd.read_pickle('data/App/app_id_nbrReviewsEmb.pkl')
        self.item_id_ReviewsEmb = pd.read_pickle('data/App/app_id_ReviewsEmb.pkl')
      
    def __getitem__(self, idx):
        user_id = self.review_df["UserID"][idx]
        item_id = self.review_df["AppID"][idx]
        y = self.review_df["Like"][idx]

        # user side
        # 會在 user_nbr_module 用到的資料
        user_data = {}
        user_emb = torch.tensor(self.user_id_emb[user_id], dtype=torch.float32)
        # 因為最外面是 list，所以要先轉成 ndarray
        user_itemEmb = torch.tensor( np.array(self.user_interacted_appEmb[user_id]), dtype=torch.float32 )
        user_nbr_revEmb = torch.tensor( np.array(self.user_id_nbrReviewsEmb[user_id]), dtype=torch.float32 )
        user_revEmb = torch.tensor( np.array(self.user_id_ReviewsEmb[user_id]), dtype=torch.float32 )
        # 包在一起，code 看起來比較簡潔
        user_data['user_id'], user_data['user_emb'], user_data['user_itemEmb'] = user_id, user_emb, user_itemEmb
        user_data['user_nbr_revEmb'], user_data['user_revEmb'] = user_nbr_revEmb, user_revEmb

        # item side
        # 會在 item_nbr_module 用到的資料
        item_data = {}
        item_emb = torch.tensor(self.item_id_emb[item_id], dtype=torch.float32)
        item_userEmb = torch.tensor( np.array(self.item_interacted_userEmb[item_id]), dtype=torch.float32 )
        item_nbr_revEmb = torch.tensor( np.array(self.item_id_nbrReviews_emb[item_id]), dtype=torch.float32 )
        item_revEmb = torch.tensor( np.array(self.item_id_ReviewsEmb[item_id]), dtype=torch.float32 )
        # 包在一起，code 看起來比較簡潔
        item_data['item_id'], item_data['item_emb'], item_data['item_userEmb'] = item_id, item_emb, item_userEmb
        item_data['item_nbr_revEmb'], item_data['item_revEmb'] = item_nbr_revEmb, item_revEmb
        
        return user_data, item_data, y

    def __len__(self):
        return len(self.review_df)