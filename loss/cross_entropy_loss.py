import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, eps=0.1, use_gpu=True):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, logit, y):
        """
        Args:
            logit: prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            y: ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        log_probs = self.logsoftmax(logit)
        zeros = torch.zeros(log_probs.size())
        y = zeros.scatter_(1, y.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            y = y.cuda()
        return (-y * log_probs).mean(0).sum()