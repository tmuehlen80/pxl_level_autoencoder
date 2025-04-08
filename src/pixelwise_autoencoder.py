import torch
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision as tv
import seaborn as sns
import matplotlib.pyplot as plt


# show some images:
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from sklearn.cluster import KMeans

import sklearn
from sklearn.decomposition import PCA


from src.osm.overpass import get_overpass_data

#from __future__ import annotations
from src.satvision.dataset import SentinelOsmDataset
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import torch
from tqdm import tqdm

from src.satvision.dataset import CNT_COLS, create_sentinel_dataset
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
# https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py#L122
weights = FCN_ResNet50_Weights.DEFAULT
#transforms = weights.transforms(resize_size=None)

#model = fcn_resnet50(weights=weights, progress=False)
#model = model.eval()
from torch.nn import Conv2d


class pxl_ae_model(nn.Module):
    def __init__(self, emb_dim:int = 10):
        super(pxl_ae_model, self).__init__()
        weights = FCN_ResNet50_Weights.DEFAULT
        self.emb_dim = emb_dim
        self.enc = fcn_resnet50(weights=weights, progress=False)
        self.dec = fcn_resnet50(weights=weights, progress=False)
        self.enc.classifier[4] = Conv2d(512, self.emb_dim, kernel_size=(1, 1), stride=(1, 1))
        self.enc.aux_classifier[4] = Conv2d(256, self.emb_dim, kernel_size=(1, 1), stride=(1, 1))
        self.dec.backbone.conv1 = Conv2d(self.emb_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.dec.classifier[4] = Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))
        self.dec.aux_classifier[4] = Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x):
        emb = self.enc(x)['out']
        recon = self.dec(emb)
        return recon['out'], emb
        

class CombinedLoss(nn.Module):
    def __init__(self, emb_dim:int = 10, n_clusters:int = 40, alpha:float = 0.5):
        super(CombinedLoss, self).__init__()
        self.loss_recon = nn.MSELoss()
        self.loss_kmeans = nn.MSELoss()
        self.emb_dim = emb_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
    def forward(self, x, recon, emb, kmeans_centroids):
        # calculate the reconstruction loss
        l_recon = self.loss_recon(x, recon)  
        # calculate the kmeans loss
        aux = emb.permute(0, 2, 3, 1).flatten(1, 2).unsqueeze(-1).expand(-1, -1, -1, self.n_clusters)
        distances = torch.square(aux - kmeans_centroids.T).sum(dim = 2)
        rec_distances = torch.div(1, distances)
        soft_arg_min = torch.nn.functional.softmax(rec_distances, dim = 2)
        dist_sargmin_sum = (soft_arg_min * distances).sum(dim = 2)
        l_kmeans = self.loss_kmeans(dist_sargmin_sum, torch.zeros_like(dist_sargmin_sum))
        return self.alpha * l_recon + (1 - self.alpha) * l_kmeans


class LossKmeans(nn.Module):
    def __init__(self, emb_dim:int = 10, n_clusters:int = 40):
        super(LossKmeans, self).__init__()
        self.loss_kmeans = nn.MSELoss()
        self.emb_dim = emb_dim
        self.n_clusters = n_clusters        
    def forward(self, emb, kmeans_centroids):
        # calculate the reconstruction loss
        #l_recon = self.loss_recon(x, recon)        
        # calculate the kmeans loss
        aux = emb.permute(0, 2, 3, 1).flatten(1, 2).unsqueeze(-1).expand(-1, -1, -1, self.n_clusters)
        distances = torch.square(aux - kmeans_centroids.T).sum(dim = 2)
        rec_distances = torch.div(1, distances)
        soft_arg_min = torch.nn.functional.softmax(rec_distances, dim = 2)
        dist_sargmin_sum = (soft_arg_min * distances).sum(dim = 2)
        l_kmeans = self.loss_kmeans(dist_sargmin_sum, torch.zeros_like(dist_sargmin_sum))

        return l_kmeans
# create dataset:
transform_img =  transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

ds = SentinelOsmDataset('/home/tmuehlen/repos/geonify/mvp/data/osm_sentinel_res_10_width_1000/data/osm_sentinel_res_10_width_1000', transform_img=transform_img)
ds_train, ds_val = torch.utils.data.random_split(ds, [0.85, 0.15])
ds.__len__()

# create dataloader
dl_train = DataLoader(ds_train, batch_size= 32, num_workers=12)
dl_val = DataLoader(ds_val, batch_size= 32, num_workers=12)
batch = next(iter(dl_train))
print("img batch shape:")
print(batch["image"].shape)
print("label batch shape:")
print(batch["label"].shape)


emb_dim = 24
model = pxl_ae_model(emb_dim=emb_dim)

batch = next(iter(dl_train))
batch.keys()
output = model(batch['image'])
print(output[1].shape, output[0].shape)

optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.MSELoss() # standard reconstruction loss


_ = model.cuda()



# standard pytorch training loop:
n_epochs = 20
best_test_loss = 100000
for epoch in range(n_epochs):
    running_loss = []
    for i, batch in enumerate(dl_train):
        optimizer.zero_grad()
        recon, emb = model(batch["image"].cuda())
        loss = criterion(recon,  batch["image"].cuda())
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        if (i % 30 == 0) & (i > 0):
            print(f"epoch {epoch}, batch {i} out of {len(dl_train)}, training loss: {pd.Series(running_loss).mean()}")
            test_loss = []
            _ = model.eval()
            for test_batch in tqdm(dl_val):
                with torch.no_grad():
                    recon, emb = model(test_batch["image"].cuda())
                    loss = criterion(recon,  test_batch["image"].cuda())
                    test_loss.append(loss.item())
            print(f"test eval: loss = {pd.Series(test_loss).mean()}")
            if pd.Series(test_loss).mean() < best_test_loss:
                best_test_loss = pd.Series(test_loss).mean()
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch
                }
                filepath = f'/home/tmuehlen/repos/geonify/mvp/data/checkpoints/pixelwise_ae/double_FCN_checkpoint_epoch_emb_dim_{emb_dim}_{epoch}_batch_{i}_test_loss_{np.round(best_test_loss, 3)}.ckpt'
                torch.save(checkpoint, filepath)
                best_test_loss = pd.Series(test_loss).mean()
                print('checkpoint saved')
            _ = model.train()


os.getcwd()

