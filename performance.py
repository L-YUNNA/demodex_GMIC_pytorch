import os
import torch
import numpy as np
from sklearn.metrics import *

class Performance():    # 나중에 perf_CI 상속받는 것도 시도해보자
    
    def get_pred(self, path, count_num, threshold):
        predictions = []
    
        for i in range(int(count_num)):
            batch_name = 'fusion_tensor_{}.pt'.format(str(i))
            batch = torch.load(os.path.join(path, batch_name))
            pred = prob_to_pred(batch, threshold)
            predictions.append(pred)
    
        preds = sum(predictions, [])
        return np.array(preds)
    
    
    def get_target(self, path, count_num):
        targets = []
    
        for i in range(int(count_num)):
            target_name = 'target_tensor_{}.pt'.format(str(i))
            target = torch.load(os.path.join(path, target_name))
            target = target.cpu().numpy()
            target = target.tolist()
            target = sum(target, [])
            targets.append(target)
    
        labels = sum(targets, [])
        return np.array(labels)
    
    
    def get_prob(self, path, count_num):
        probability = []
    
        for i in range(int(count_num)):
            output = 'fusion_tensor_{}.pt'.format(str(i))
            batch = torch.load(path + output)
    
            proba = batch.cpu().detach().numpy()
            proba = proba.tolist()
            proba = sum(proba, [])
            probability.append(proba)
    
        probs = sum(probability, [])
        return np.array(probs)
    
    
    def get_performance(self, path, count_num):
        y_true = self.get_target(path, count_num)
        y_pred = self.get_pred(path, count_num, threshold=0.5)
        y_prob = self.get_prob(path, count_num)
    
        print("[Summary]\n", classification_report(y_true, y_pred, digits=5))
        print("[AUC] : ", roc_auc_score(y_true, y_prob))
    
        return y_true, y_pred, y_prob


def prob_to_pred(y_prob, threshold):
    y_pred = []
    for prob in y_prob:
        if prob > threshold: pred = 1.0
        else: pred = 0.0
        y_pred.append(pred)
    return y_pred