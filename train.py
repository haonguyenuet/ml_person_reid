import torch

import models
import losses
import dataset

import argparse


def train_epoch(epoch, model, trainloader, criterion, optimizer, print_freq = 10, use_gpu = True):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for batch_index, data in enumerate(trainloader):
      loss, acc = forward_backward(model, data, criterion, optimizer, use_gpu)

      running_loss += loss
      running_acc += acc
      if (batch_index + 1) % print_freq == 0:
        print('[%d, %5d]\t loss: %.3f\t accuracy: %.3f' 
              %(epoch + 1, batch_index + 1, running_loss / print_freq, running_acc / print_freq))
        running_loss = 0.0
        running_acc  = 0.0

def forward_backward(model, data, criterion, optimizer, use_gpu = True):
    imgs, pids = parse_data_for_train(data)
    
    if use_gpu:
        imgs, pids = imgs.cuda(), pids.cuda()

    logits = model(imgs)
    loss = criterion(logits, pids)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, prediction = torch.max(logits, dim=1)
    acc = (prediction == pids).sum() * 100 / len(pids)

    return loss.item(), acc.item()


def parse_data_for_train(data):
    imgs = data['img']
    pids = data['pid']
    return imgs, pids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Path to store data")
    parser.add_argument("--dataset_name", type=str,
                        default="market1501", help="dataset name")
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

    data_preparer = dataset.DataPreparer(
        root=args.root,
        dataset_name=args.dataset_name,
        batch_size_train=args.batch_size_train,
        workers=args.workers,
    )

    num_classes = data_preparer.num_train_pids

    model = models.OSNet(num_classes=num_classes)
    if args.use_gpu:
        model = model.cuda()
    
    criterion = losses.CrossEntropyLoss(num_classes =num_classes, use_gpu= args.use_gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.max_epoch):
        train_epoch(    
            epoch = epoch,
            model=model,
            trainloader = data_preparer.trainloader,
            criterion = criterion,
            optimizer = optimizer,
            print_freq= args.print_freq,
            use_gpu= args.use_gpu
        )
        # torch.save(model.state_dict(), './pretrained_model.pth')
