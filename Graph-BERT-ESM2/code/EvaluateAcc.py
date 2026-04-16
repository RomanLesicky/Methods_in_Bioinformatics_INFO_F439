'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: MIT

# Code originally modified so now CUDA tensors are correctly not being fed into sklearn but rather correct numpy arrays 

import torch
from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score


class EvaluateAcc(evaluate):
    data = None

    def evaluate(self):
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        if torch.is_tensor(true_y):
            true_y = true_y.detach().cpu().numpy()
        if torch.is_tensor(pred_y):
            pred_y = pred_y.detach().cpu().numpy()

        return accuracy_score(true_y, pred_y) 
        