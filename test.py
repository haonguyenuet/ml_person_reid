import torch

import numpy as np
import argparse

import models
import metrics
import dataset


def evaluate(model, query_loader=None, gallery_loader=None, use_gpu=True):
    model.eval()

    q_features, q_pids, q_camids = feature_extraction(
        model,
        query_loader,
        use_gpu
    )
    g_features, g_pids, g_camids = feature_extraction(
        model,
        gallery_loader,
        use_gpu
    )

    print('Computing feature distance ...')
    print('q_features(num_q, num_pids)')
    print('g_features(num_g, num_pids)')
    print('dismat(num_q, num_g)')
    distmat = metrics.cosine_distance(q_features, g_features)
    distmat = distmat.numpy()

    print('Computing CMC and mAP ...')
    cmc, mAP = metrics.eval_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
    )

    return cmc[0], mAP


def feature_extraction(model, loader, use_gpu=True):
    features_, pids_, camids_ = [], [], []
    for _, data in enumerate(loader):
        imgs, pids, camids = parse_data_for_test(data)
        if use_gpu:
            imgs = imgs.cuda()
        features = model(imgs)
        features = features.cpu().clone()
        features_.append(features)
        pids_.extend(pids)
        camids_.extend(camids)
    features_ = torch.cat(features_, 0)
    pids_ = np.asarray(pids_)
    camids_ = np.asarray(camids_)
    return features_, pids_, camids_


def parse_data_for_test(data):
    imgs = data['img']
    pids = data['pid']
    camids = data['camid']
    return imgs, pids, camids

def load_model(model, use_gpu):
    if use_gpu:
       device = torch.device('cuda:0')
       model.load_state_dict(torch.load('./pretrained_model.pth'))
       model.to(device)
    else:
       device = torch.device('cpu')
       model.load_state_dict(torch.load('./pretrained_model.pth',map_location=device))

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Path to store data")
    parser.add_argument("--dataset_name", type=str,
                        default="market1501", help="dataset name")
    parser.add_argument("--batch_size_test", type=int, default=32,
                        help="number of images in a testing batch. Default is 32")
    parser.add_argument("--workers", type=int, default=4,
                        help="number of workers. Default is 4")
    parser.add_argument("--use_gpu", type=bool, default=False,
                        help="use gpu. Default is False")

    args = parser.parse_args()

    datamanager = dataset.DataPreparer(
        root=args.root,
        dataset_name=args.dataset_name,
        batch_size_test=args.batch_size_test,
        workers=args.workers,
    )

    model = models.OSNet(num_classes=datamanager.num_train_pids)
    model = load_model(model, args.use_gpu)

    rank1, mAP = evaluate(
        model=model,
        query_loader=datamanager.testloader.query,
        gallery_loader=datamanager.testloader.gallery,
        use_gpu=args.use_gpu
    )
    print("Rank 1: ", rank1)
    print("mAP: ", mAP)
