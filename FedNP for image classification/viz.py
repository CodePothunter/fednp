import argparse
from utils import clf_dataset, split_dataset
from torchvision import datasets, transforms
import torch
from utils import clf_dataset, split_dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import math
from libs import torch as libs
from models.resnet import ResNet18

libs.init_Seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--user', default='Dylon', type=str)
parser.add_argument('--experiment', default='NP', type=str)
parser.add_argument('--date', default=local_time, type=str)
parser.add_argument('--description', default='non-iid', type=str)
# training data
parser.add_argument('--root', default='path to training set', type=str)
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--data_dist', default='noniid', type=str)
parser.add_argument('--num_workers', default=4, type=int)
# Training Information
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--K', default=10, type=int, help="number of clients")
parser.add_argument('--wd', default=1e-5, type=float)
parser.add_argument('--mu', default=0.01, type=float, help="loss factor")
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--local_epochs', default=10, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--save_freq', default=50, type=int)
parser.add_argument('--beta', default=0.5, type=float, help="parameter of Dirichlet distribution")
parser.add_argument('--model_path', default='path to trained model', type=str)


args = vars(parser.parse_args())

train_dataset, test_dataset, num_classes = clf_dataset(args)
train_datasets, weights = split_dataset(args, train_dataset)
train_loaders = []

for i in range(args['K']):
    train_loaders.append(torch.utils.data.DataLoader(dataset=train_datasets[i], batch_size=128, num_workers=4, pin_memory=True, shuffle=False))


model = ResNet18(num_classes=100).cuda()
model.load_state_dict(torch.load(args['model_path'])['state_dict'])
model.eval()
features = []
labels = []
clients = []
for k in range(args['K']):
    with torch.no_grad():
        for X, y in train_loaders[k]:
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            idx = y < 10
            if idx.sum() > 0:
                X = X[idx]
                y = y[idx]
                y_pred, fx = model(X)
                features.append(fx)
                labels.append(y)
                clients.append(torch.ones_like(y) * k)

yy = torch.cat(labels)
a = torch.cat(features)
cc = torch.cat(clients)

center = model.linear.weight.detach().clone()
a = torch.cat([a, center]).cpu().numpy()
yy = torch.cat([yy, torch.arange(num_classes).cuda()]).cpu().numpy()
cc = torch.cat([cc, torch.ones(num_classes).cuda()*10])
a = a / np.linalg.norm(x=a, ord=2, axis=1, keepdims=True)


tsne = TSNE(n_components=2)
b = tsne.fit_transform(a)

colormap = [[0.12156862745098039, 0.4666666666666667, 0.7058823529411765],
[1.0, 0.4980392156862745, 0.054901960784313725],
[0.17254901960784313, 0.6274509803921569, 0.17254901960784313],
[0.8392156862745098, 0.15294117647058825, 0.1568627450980392],
[0.5803921568627451, 0.403921568627451, 0.7411764705882353],
[0.5490196078431373, 0.33725490196078434, 0.29411764705882354],
[0.8901960784313725, 0.4666666666666667, 0.7607843137254902],
[0.4980392156862745, 0.4980392156862745, 0.4980392156862745],
[0.7372549019607844, 0.7411764705882353, 0.13333333333333333],
[0.09019607843137255, 0.7450980392156863, 0.8117647058823529]]

fig, axes = plt.subplots(1, 1, figsize=(5, 5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0, hspace=0.)
axes.scatter(b[:5000, 0], b[:5000, 1], c=[colormap[t] for t in yy[:5000].tolist()], s=1)
axes.set_yticks([])
axes.set_xticks([])
axes.set_xlim([-75, 75])
axes.set_ylim([-75, 75])


p = []
for y in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    p.append(-2 * math.log(y) * 2000)

for i, g in enumerate(p):
    axes.scatter(b[5000:5010, 0], b[5000:5010, 1], c=[colormap[t] for t in yy[5000:5010].tolist()], s=g+100, alpha=0.1)
plt.savefig('npn_dist.pdf', bbox_inches = 'tight')


fig, axes = plt.subplots(2, 5, figsize=(15, 6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0, hspace=0.)

for k in range(10):
    idx = (cc == k).cpu().numpy()
    axes[k//5][k%5].scatter(b[:, 0][idx], b[:, 1][idx], c=[colormap[t] for t in yy[idx].tolist()], s=5)
    axes[k//5][k%5].set_yticks([])
    axes[k//5][k%5].set_xticks([])
    axes[k//5][k%5].set_title(f'client {k}', x = 0.78, y=0.8, fontdict={'weight':'bold','size': 13})
plt.savefig('tsne_fednp.pdf', bbox_inches = 'tight')
