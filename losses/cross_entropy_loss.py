import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, use_gpu=True, label_smooth = True):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = 0.1 if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, targets):
        """
        Args:
            logits: prediction matrix with shape (batch_size, num_classes).
            targets: Each position contains the true label index with shape (batch_size).
        """
        log_probs = self.logsoftmax(logits)
        zeros = torch.zeros(log_probs.size())
        y = zeros.scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            y = y.cuda()
        y = (1 - self.eps) * y + self.eps / self.num_classes
        return (-y * log_probs).mean(0).sum()