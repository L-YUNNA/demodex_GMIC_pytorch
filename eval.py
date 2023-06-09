# Copyright (C) 2020 Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
# Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of GMIC.
#
# GMIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# GMIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""
Script that executes the model pipeline.
"""

import argparse
import random
import shutil
import pickle
import tqdm
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.modeling import gmic as gmic
from src.data_loading.cv_preprocessing import *
from src.data_loading.data_loader import *
from performance import *



def main():
    # retrieve command line arguments
    parser = argparse.ArgumentParser(description='Run GMIC on the sample data')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--device-type', default="gpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--best-fold', type=int, help='number of best fold number (1~10)')

    global args
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)

    parameters = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        # model related hyper-parameters
        "cam_size": (92, 60),
        "K": 6,
        "crop_shape": (256, 256),
        "post_processing_dim":512,
        "num_classes":1,
        "percent_t":0.05,
        "use_v1_global":True,
    }

    with open('./checkpoints/CV/foldperf.pkl', 'rb') as f:
        foldperf = pickle.load(f)

    img_dir = os.path.join(args.data_path, 'images')
    df = pd.read_excel(args.data_path + '/data_list/train_case1_230517_ML.xlsx', header=0, engine='openpyxl')
    eval_df = pd.read_excel(args.data_path + '/data_list/test_case1_230517_ML.xlsx', header=0, engine='openpyxl')

    # best fold에 사용된 train and valid data index
    train_idx = foldperf['fold{}'.format(args.best_fold)]['train']['train_idx'][0]
    valid_idx = foldperf['fold{}'.format(args.best_fold)]['valid']['valid_idx'][0]
    re_train_df = df.loc[train_idx].reset_index(drop=True)
    re_valid_df = df.loc[valid_idx].reset_index(drop=True)

    # feature scailing, 학습에 사용된 train으로 fit.transform
    scaler = StandardScaler()
    _, _, scaled_eval_df = scaled_datasets(re_train_df, re_valid_df, eval_df, scaler, continuous_feat=['Age',
                                                                                                       'Eo_count',
                                                                                                       'Total_IgE',
                                                                                                       'ECP'])
    size = (1920, 2944)
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    tf = transforms.Compose([transforms.ToTensor(),
                             normalize])

    eval_dataset = CombineDataset(scaled_eval_df, 'img_name', 'cls_adj', img_dir, input_size=size, transform=tf)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=True)

    # create model
    device = torch.device("cuda:{}".format(parameters["gpu_number"]))
    model = gmic.GMIC(parameters)
    model = model.to(device)

    # apply TL to globalnet, localnet
    use_TL(model.ds_net)
    use_TL(model.dn_resnet)

    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # load best fold model
    resume = './checkpoints/CV/Fold{}_model_best.pth.tar'.format(args.best_fold)
    load_best_model(resume, model, optimizer)

    evaluate(eval_loader, criterion, model, device)

    # caculate performance
    num_of_batch = len(eval_loader)   # 총 배치의 개수 (데이터 100개, batch_size=4 -> 총 배치 개수는 25개)
    output_tensor_path = './checkpoints/evaluation/'
    y_true, y_pred, y_prob = Performance().get_performance(path=output_tensor_path, count_num=num_of_batch)


def evaluate(eval_loader, criterion, model, device):

    model.eval()

    for i, (clinical_input, img_input, target) in enumerate(eval_loader):
        target = target.to(torch.float).view(len(target), 1)
        target = target.to(device)
        img_input = img_input.to(device)
        clinical_input = clinical_input.to(device)

        with torch.no_grad():
            clinical_input_var = torch.autograd.Variable(clinical_input)
            img_input_var = torch.autograd.Variable(img_input)
            target_var = torch.autograd.Variable(target)

        # output
        y_fusion, y_global, y_local, Ac = model(img_input_var, clinical_input_var)
        loss_fusion = criterion(y_fusion, target_var)

        if not os.path.exists('./checkpoints/evaluation'):
                os.mkdir('./checkpoints/evaluation')
        torch.save(y_fusion, './checkpoints/evaluation/fusion_tensor_{}.pt'.format(i))
        torch.save(target, './checkpoints/evaluation/target_tensor_{}.pt'.format(i))


def load_best_model(resume, model, optimizer):
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))


def use_TL(module_net):
    resnet18 = models.resnet18(pretrained=True)
    res18_dict = resnet18.state_dict()

    module_dict = module_net.state_dict()
    filtered_dict = {k: v for k, v in res18_dict.items() if k in module_dict}
    module_dict.update(filtered_dict)
    module_net.load_state_dict(module_dict)

    for param in module_net.parameters():
        param.requires_grad = False

def apply_fine_tune(module_net):
    layer = []
    for name, _ in module_net.named_parameters():
        layer.append(name)
    retrained_layer = layer[45:]  # layer4=45, layer3=30, layer2=15

    for name, param in module_net.named_parameters():
        if name in retrained_layer:
            param.requires_grad = True

if __name__ == "__main__":
    main()
