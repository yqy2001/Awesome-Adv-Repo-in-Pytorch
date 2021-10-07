import torch

import numpy as np
import pandas as pd
# from tsne import bh_sne
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def gen_features(net, device, dataloader):
    net.eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for _, inputs, targets in tqdm(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_np = targets.cuda().data.cpu().numpy()

            outputs, feature = net(inputs)
            outputs_np = feature.cuda().data.cpu().numpy()
            
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)
            
    targets = np.concatenate(targets_list, axis=0)
    features = np.concatenate(outputs_list, axis=0).astype(np.float64)

    return targets, features


def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets
    
    plt.clf()
    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )
    
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(save_dir, bbox_inches='tight')
    print('done!')


def tsne(net, device, dataloader, save_dir):
    targets, outputs = gen_features(net, device, dataloader)
    print("Feature get!")
    tsne_plot(save_dir, targets, outputs)