import numpy as np
import torch
from matplotlib import pyplot as plt

def samples_to_images(samples):
    samples_clone=torch.clone(samples)
    samples_clone.clamp_(min=-1, max=1)
    samples_clone.sub_(-1).div_(2)
    return samples_clone.mul(255).add_(0.5).clamp_(0, 255).permute(0,2,3,1).to("cpu", torch.uint8).numpy()

def gridplot(Nx,Ny,images):
    plt.figure(figsize=(Ny*2,Nx*2))
    for i in range(Nx):
        for j in range(Ny):
            plt.subplot(Nx,Ny,j+Ny*i+1)
            plt.imshow(images[j+Ny*i])
            plt.axis("off")

from sklearn.decomposition import PCA          
def calculate_PCA_of_w(style_gan,N_w=10000,device='cuda'):
    total_w=[]
    for _ in range(N_w//100):
        sample_z = torch.randn(100, 512, device=device)
        with torch.no_grad():
            _, w = style_gan([sample_z],return_latents=True)
        total_w.append(w)
    total_ws=torch.cat(total_w).cpu().numpy()
    total_ws=total_ws[:,0,:]
    
    pca = PCA(n_components=512)
    ws_pca=pca.fit_transform(total_ws)
    w_90_in_pca=np.percentile(ws_pca,90,axis=0)
    w_10_in_pca=np.percentile(ws_pca,10,axis=0)
    
    pca_of_w=pca.components_
    pca_of_w_tensor=torch.tensor(pca_of_w.T).to(device)
    
    return pca_of_w_tensor,w_10_in_pca,w_90_in_pca


def edit_gan(sample_z,style_gan,pca_of_w,w_min_in_pca,w_max_in_pca,edit_axis=0,edit_num=10):
    with torch.no_grad():
        ori_gal, w = style_gan([sample_z],return_latents=True,randomize_noise=False)
        
    w_in_pca=torch.matmul(w,pca_of_w)
    
    gals=torch.zeros(edit_num,3,256,256)
    for idx,w_value in enumerate(np.linspace(w_min_in_pca[edit_axis],w_max_in_pca[edit_axis],edit_num)):
        w_in_pca[0,:,edit_axis]=w_value
        new_w=torch.matmul(w_in_pca,pca_of_w.T)
        with torch.no_grad():
            gal,_= style_gan([new_w],input_is_latent=True,randomize_noise=False)
        gals[idx]=gal
    
    return ori_gal,gals
    