import torch

import models
import losses
from data import DataManager

import argparse


class Trainer(object):
    def __init__(self, model, dm, criterion, optimizer, use_gpu, args: argparse.Namespace):
        self.args = args
        self.model = model
        self.train_loader = dm.train_loader()
        self.criterion, self.optimizer = criterion, optimizer
        self.use_gpu = use_gpu

    def do_train(self):
        for epoch in range(args.max_epoch):
            self.train_epoch(epoch=epoch)
            torch.save(self.model.state_dict(), './pretrained_model.pth')

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0

        for batch_idx, data in enumerate(self.train_loader):
            loss, acc = self.forward_backward(data)

            running_loss += loss
            running_acc += acc
            if (batch_idx + 1) % args.print_freq == 0:
                print('[%d, %5d]\t loss: %.3f\t accuracy: %.3f'
                      % (epoch + 1, batch_idx + 1, running_loss / args.print_freq, running_acc / args.print_freq))
                running_loss = 0.0
                running_acc = 0.0

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        logits = self.model(imgs)
        loss = self.criterion(logits, pids)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        _, prediction = torch.max(logits, dim=1)
        acc = (prediction == pids).sum() * 100 / len(pids)

        return loss.item(), acc.item()

    def parse_data_for_train(self, data):
        imgs = data['img']
        pids = data['pid']
        return imgs, pids

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Trainer")
        parser.add_argument("--max_epoch", type=int, default=20)
        parser.add_argument("--print_freq", type=int, default=10)
        return parent_parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = DataManager.add_model_specific_args(parser)
    parser = Trainer.add_model_specific_args(parser)
    parser.add_argument("--use_gpu", type=bool, default=False)

    args = parser.parse_args()

    dm = DataManager(args)

    model = models.OSNet(num_classes=dm.num_train_classes)
    if args.use_gpu:
        model = model.cuda()

    criterion = losses.CrossEntropyLoss(num_classes=dm.num_train_classes, use_gpu=args.use_gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    trainer = Trainer(model, dm, criterion, optimizer, args.use_gpu, args)
    trainer.do_train()
