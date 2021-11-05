import torch

import numpy as np
import argparse
import os

import model as nets
import metrics
from dataset.datamanager import DataManager

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


def feature_extraction(model, loader, use_gpu):
    features_, pids_, camids_ = [], [], []
    for _, data in enumerate(loader):
        imgs, pids, camids = parse_data_for_test(data)
        if use_gpu:
            imgs = imgs.cuda()
        features = model(imgs)
        # features = features.cpu().clone()
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


def load_model(model, model_name, epoch_label):
    save_filename = 'model_%s.pth' % epoch_label
    save_path = os.path.join('./model', model_name, save_filename)
    model.load_state_dict(torch.load(save_path))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Path to store data")
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    parser.add_argument("--which_epoch", type=int, help="epoch label")
    parser.add_argument("--height", type=int, default=256,
                        help="target image height. Default is 256")
    parser.add_argument("--width", type=int, default=128,
                        help="target image width. Default is 128")
    parser.add_argument("--transforms", type=str, default='random_flip',
                        help="transformations applied to model training. Default is random_flip")
    parser.add_argument("--batch_size_test", type=int, default=32,
                        help="number of images in a testing batch. Default is 32")
    parser.add_argument("--workers", type=int, default=4,
                        help="number of workers. Default is 4")
    parser.add_argument("--use_gpu", type=bool, default=False,
                        help="use gpu. Default is False")

    args = parser.parse_args()

    datamanager = DataManager(
        root=args.root,
        dataset_name=args.dataset_name,
        height=args.height,
        width=args.width,
        transforms=args.transforms,
        batch_size_test=args.batch_size_test,
        workers=args.workers,
    )

    model = nets.OSNet(num_classes=datamanager.num_train_pids)
    model = load_model(model, 'osnet', args.which_epoch)

    rank1, mAP = evaluate(
        model=model,
        query_loader=datamanager.testloader.query,
        gallery_loader=datamanager.testloader.gallery,
        use_gpu=args.use_gpu
    )
