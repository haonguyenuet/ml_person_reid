import torch
from model.osnet import OSNet
from loss.cross_entropy_loss import CrossEntropyLoss
from dataset.datamanager import DataManager

from metrics.accuracy import accuracy

import argparse


def parse_data_for_train(data):
    imgs = data['img']
    pids = data['pid']
    return imgs, pids


def forward_backward(model, data, datamanager, use_gpu=True):
    imgs, pids = parse_data_for_train(data)

    if use_gpu:
        imgs, pids = imgs.cuda(), pids.cuda()

    criterion = CrossEntropyLoss(
        num_classes=datamanager.num_train_pids, use_gpu=use_gpu)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    outputs = model(imgs)
    loss = criterion(outputs, pids)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = accuracy(outputs, pids)[0]

    return loss.item(), acc.item()


def train_epoch(model, datamanager, epoch, print_freq, use_gpu):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for batch_index, data in enumerate(datamanager.trainloader):
        loss, acc = forward_backward(model, data, datamanager, use_gpu=use_gpu)

        running_loss += loss
        running_acc += acc
        if (batch_index + 1) % print_freq == 0:
            print('[%d, %5d]\t loss: %.3f\t accuracy: %.3f'
                  % (epoch + 1, batch_index + 1, running_loss / print_freq, running_acc / print_freq))
            running_loss = 0.0
            running_acc = 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Path to store data")
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    parser.add_argument("--height", type=int, default=256,
                        help="target image height. Default is 256")
    parser.add_argument("--width", type=int, default=128,
                        help="target image width. Default is 128")
    parser.add_argument("--transforms", type=str, default='random_flip',
                        help="transformations applied to model training. Default is random_flip")
    parser.add_argument("--batch_size_train", type=int, default=32,
                        help="number of images in a training batch. Default is 32")
    parser.add_argument("--workers", type=int, default=4,
                        help="number of workers. Default is 4")
    parser.add_argument("--max_epoch", type=int, default=20,
                        help="number of epochs. Default is 20")
    parser.add_argument("--print_freq", type=int, default=10,
                        help="printer frequency. Default is 10")
    parser.add_argument("--use_gpu", type=bool, default=False,
                        help="use gpu. Default is False")

    args = parser.parse_args()

    datamanager = DataManager(
        root=args.root,
        dataset_name=args.dataset_name,
        height=args.height,
        width=args.width,
        transforms=args.transforms,
        batch_size_train=args.batch_size_train,
        workers=args.workers,
    )

    model = OSNet(num_classes=datamanager.num_train_pids)

    for epoch in range(args.max_epoch):
        train_epoch(
            model=model,
            datamanager=datamanager,
            epoch=epoch,
            print_freq=args.print_freq,
            use_gpu=args.use_gpu
        )
        torch.save(model.state_dict(), './model.pth')
