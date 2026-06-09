# lenet base model for Pareto MTL
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import CrossEntropyLoss


class RegressionTrain(torch.nn.Module):
  
    def __init__(self, model):
        super(RegressionTrain, self).__init__()
        
        self.model = model
        self.ce_loss = CrossEntropyLoss()
    
    def forward(self, x, ts):
        n_tasks = 3
        ys = self.model(x)
     
        task_loss = []
        for i in range(n_tasks):
            task_loss.append( self.ce_loss(ys[i], ts[:,i]) )
        task_loss = torch.stack(task_loss)

        return task_loss



        
    
   