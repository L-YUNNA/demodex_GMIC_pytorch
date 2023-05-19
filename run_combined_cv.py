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

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from modeling import gmic as gmic
from data_loading.cv_preprocessing import *
from data_loading.data_loader import *



def train(train_loader, model, criterion, optimizer, epoch, device):

    model.train()

    history = {'true':[], 'prob':[], 'pred':[], 'loss':[]}

    for i, (clinical_input, img_input, target) in enumerate(train_loader):
        target = target.to(torch.float).view(len(target),1)
        target = target.to(device)
        img_input = img_input.to(device)
        clinical_input = clinical_input.to(device)

        target_var = torch.autograd.Variable(target)
        img_input_var = torch.autograd.Variable(img_input)
        clinical_input_var = torch.autograd.Variable(clinical_input)

        # output
        y_fusion, y_global, y_local, Ac = model(img_input_var, clinical_input_var)
        loss_global = criterion(y_global, target_var)
        loss_local = criterion(y_local, target_var)
        loss_fusion = criterion(y_fusion, target_var)

        # measure accuracy and record loss
        beta = 0.00001
        loss_cam = cam_loss(Ac, beta)
        loss = loss_global + loss_local + loss_fusion + loss_cam
        #loss = loss_global + loss_local + loss_fusion

        history['loss'].append(loss.data.cpu().numpy())

        prob_numpy = y_fusion.data.cpu().numpy()
        target_numpy = target_var.data.cpu().numpy()

        append_val_from_batch(prob_numpy, history['prob'])
        append_val_from_batch(target_numpy, history['true'])

        # compute gradient and do ADAM step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: [{0}] - batch: [{1}/{2}]\t'
              'Loss: {3}'.format(epoch, i, len(train_loader), loss))

    pred = prob_to_pred(history['prob'])
    history['pred'].append(pred)

    acc = accuracy_score(history['true'], pred)
    auc = roc_auc_score(history['true'], history['prob'])
    print('Epoch: [{0}]\t'
          'Acc : {1}, AUC : {2}, Loss : {3}'
          .format(epoch, acc, auc, np.mean(history['loss'])))

    return acc, auc, np.mean(history['loss']), history


def validate(valid_loader, model, criterion, device, fold, mode):

    model.eval()

    history = {'true': [], 'prob': [], 'pred': [], 'loss': []}

    for i, (clinical_input, img_input, target) in enumerate(valid_loader):
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
        loss_global = criterion(y_global, target_var)
        loss_local = criterion(y_local, target_var)
        loss_fusion = criterion(y_fusion, target_var)

        if mode=="test":
            if not os.path.exists('./checkpoints/CV/output'):
                os.mkdir('./checkpoints/CV/output')
            torch.save(y_fusion, './checkpoints/CV/output/Fold{}_fusion_tensor_{}.pt'.format(fold + 1, i))
            torch.save(y_global, './checkpoints/CV/output/Fold{}_global_tensor_{}.pt'.format(fold + 1, i))
            torch.save(y_local, './checkpoints/CV/output/Fold{}_local_tensor_{}.pt'.format(fold + 1, i))
            torch.save(target, './checkpoints/CV/output/Fold{}_target_tensor_{}.pt'.format(fold + 1, i))

        # measure accuracy and record loss
        beta = 0.00001
        loss_cam = cam_loss(Ac, beta)
        loss = loss_global + loss_local + loss_fusion + loss_cam
        #loss = loss_global + loss_local + loss_fusion

        history['loss'].append(loss.data.cpu().numpy())

        prob_numpy = y_fusion.data.cpu().numpy()
        target_numpy = target_var.data.cpu().numpy()

        append_val_from_batch(prob_numpy, history['prob'])
        append_val_from_batch(target_numpy, history['true'])

        print('Test batch: [{0}/{1}]\t'
              'Loss: {2}'.format(i, len(valid_loader), loss))

    pred = prob_to_pred(history['prob'])
    history['pred'].append(pred)

    acc = accuracy_score(history['true'], pred)
    auc = roc_auc_score(history['true'], history['prob'])
    print('Test ACC : {0}, Test AUC : {1}, Test Loss : {2}'
          .format(acc, auc, np.mean(history['loss'])))

    return acc, auc, np.mean(history['loss']), history


def append_val_from_batch(batch, list_name):
    for i in range(len(batch)):
        value = batch[i, 0]
        list_name.append(value)


def prob_to_pred(y_prob):
    y_pred = []
    for prob in y_prob:
        if prob > 0.5: pred = 1.0
        else: pred = 0.0
        y_pred.append(pred)
    return y_pred



def main():
    # retrieve command line arguments
    parser = argparse.ArgumentParser(description='Run GMIC on the sample data')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--device-type', default="gpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    # parser.add_argument('--ngpu', default=3, type=int, metavar='G', help='number of gpus to use')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W')
    parser.add_argument('--epochs', default=90, type=int, metavar='N')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')


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

    img_dir = os.path.join(args.data_path, 'images')
    df = pd.read_excel(args.data_path + '/data_list/train_case1_230517_ML.xlsx', header=0, engine='openpyxl')
    test_df = pd.read_excel(args.data_path + '/data_list/test_case1_230517_ML.xlsx', header=0, engine='openpyxl')

    size = (1920, 2944)
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 추후 수정, mean & std로..
    tf = transforms.Compose([transforms.ToTensor(),
                             normalize])

    Y = df['cls_adj']
    
    scaler = StandardScaler()
    splits = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
    foldperf = {}

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(df)), Y)):
        print('***** Fold {} *****'.format(fold + 1))
        best_auc = 0

        # split train, valid
        re_train_df = df.loc[train_idx].reset_index(drop=True)
        re_valid_df = df.loc[val_idx].reset_index(drop=True)

        # change the valid set's class ratio to  1:1
        cls0_indices = np.where(re_valid_df['cls_adj'] == 0)[0]
        cls1_indices = np.where(re_valid_df['cls_adj'] == 1)[0]

        # max_samples = max(len(cls0_indices), len(cls1_indices))
        # min_samples = min(len(cls0_indices), len(cls1_indices))
        # ratio = int(max_samples / min_samples)  # 3배 (대략 0:71, 1:21)

        cls1_valid_df = re_valid_df.loc[cls1_indices]
        re_valid_df = pd.concat([re_valid_df,
                                 cls1_valid_df,
                                 cls1_valid_df]).reset_index(drop=True)  # 대략 0:71, 1:63

        # clinical data feature scaling
        scaled_train_df, scaled_valid_df, scaled_test_df = scaled_datasets(re_train_df,
                                                                           re_valid_df,
                                                                           test_df,
                                                                           scaler, continuous_feat=['Age',
                                                                                                  'case1_Eo_count',
                                                                                                  'case1_Total_IgE',
                                                                                                  'case1_ECP'])
        fold_train_dataset = CombineDataset(scaled_train_df,
                                            'img_name', 'cls_adj', img_dir, input_size=size, transform=tf)
        fold_valid_dataset = CombineDataset(scaled_valid_df,
                                            'img_name', 'cls_adj', img_dir, input_size=size, transform=tf)
        test_dataset = CombineDataset(scaled_test_df,
                                      'img_name', 'cls_adj', img_dir, input_size=size, transform=tf)

        # augmentation
        aug_list = ['AWB', 'jitter', 'gnoise', 'ro', 'shear', 'hflip', 'vflip']
        aug_fold_train_dataset = augmentation(scaled_train_df, fold_train_dataset, 'cls_adj', img_dir, size, tf, aug_list)

        train_loader = DataLoader(aug_fold_train_dataset,
                                  batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(fold_valid_dataset,
                                  batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True, drop_last=True)

        # create model
        device = torch.device("cuda:{}".format(parameters["gpu_number"]))
        model = gmic.GMIC(parameters)

        # apply TL to globalnet, localnet and fine-tune
        use_TL(model.ds_net)
        use_TL(model.dn_resnet)
        #apply_fine_tune(model.dn_resnet)

        # CHECK, 생략 가능
        g = [print("Global :", para.requires_grad) for para in model.ds_net.parameters()]
        l = [print("Local :", local_para.requires_grad) for local_para in model.dn_resnet.parameters()]

        #model = nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        model = model.to(device)
        criterion = nn.BCELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                    args.lr,
                                    weight_decay=args.weight_decay)

        train_perf = {'train_idx': [], 'train_acc': [], 'train_auc': [], 'train_loss': [], 'history': []}
        valid_perf = {'valid_idx': [], 'valid_acc': [], 'valid_auc': [], 'valid_loss': [], 'history': []}
        train_perf['train_idx'].append(train_idx)
        valid_perf['valid_idx'].append(val_idx)

        for epoch in range(args.epochs):
            # train
            train_acc, train_auc, train_loss, train_history = train(train_loader,
                                                                    model,
                                                                    criterion,
                                                                    optimizer,
                                                                    epoch,
                                                                    device,)
            train_perf['train_acc'].append(train_acc)
            train_perf['train_auc'].append(train_auc)
            train_perf['train_loss'].append(train_loss)
            train_perf['history'].append(train_history)

            # validate
            valid_acc, valid_auc, valid_loss, history = validate(valid_loader,
                                                                model,
                                                                criterion,
                                                                device,
                                                                fold,
                                                                mode='valid')

            valid_perf['valid_acc'].append(valid_acc)
            valid_perf['valid_auc'].append(valid_auc)
            valid_perf['valid_loss'].append(valid_loss)
            valid_perf['history'].append(history)

            is_best = valid_auc > best_auc
            best_auc = max(valid_auc, best_auc)
            save_checkpoint({
                'epoch':epoch+1,
                'state_dict': model.state_dict(),
                'best_auc': best_auc,
                'optimizer': optimizer.state_dict(),
            }, is_best, fold)

        print("*** Predict test_loader using the fold_best_model...")
        resume = './checkpoints/CV/Fold{}_model_best.pth.tar'.format(fold+1)
        load_best_model(resume, model, optimizer)
        test_acc, test_auc, test_loss, test_history = validate(test_loader,
                                                               model,
                                                               criterion,
                                                               device, fold, mode="test")
        test_perf = {'test_acc': test_acc, 'test_auc': test_auc, 'test_loss': test_loss, 'history': test_history}
        foldperf['fold{}'.format(fold + 1)] = {'train': train_perf, 'valid': valid_perf, 'test': test_perf}

    with open("./checkpoints/CV/foldperf.pkl", "wb") as f:
        pickle.dump(foldperf, f)


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


def cam_loss(saliency_map, beta):
    # flatten_Ac = saliency_map.abs().view(-1)
    # l1reg = 0
    # for j in range(len(flatten_Ac)):
    #     l1reg += flatten_Ac[j]
    # avg_l1reg = l1reg/len(flatten_Ac)
    avg_l1reg = saliency_map.mean(dim=-1).mean(dim=-1).mean(axis=0)[0]
    return beta * avg_l1reg


def save_checkpoint(state, is_best, fold):
    if not os.path.exists('./checkpoints/CV'):
        os.mkdir('./checkpoints/CV')

    filename = './checkpoints/CV/Fold{}_checkpoint.pth.tar'.format(fold+1)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/CV/Fold{}_model_best.pth.tar'.format(fold+1))


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
