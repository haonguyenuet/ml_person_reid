import torch

import numpy as np
import argparse

import models
import metrics
from data import DataManager


class Tester(object):
    def __init__(self, model, dm, use_gpu):
        self.model = model
        self.test_loader = dm.test_loader()
        self.use_gpu = use_gpu

    def evaluate(self):
        self.model.eval()

        q_features, q_pids, q_camids = self.feature_extraction(
            self.test_loader.query,
        )
        g_features, g_pids, g_camids = self.feature_extraction(
            self.test_loader.gallery,
        )

        print('Computing feature distance ...')
        print('q_features(num_q, num_pids)')
        print('g_features(num_g, num_pids)')
        print('dismat(num_q, num_g)')
        distmat = metrics.cosine_distance(q_features, g_features)
        distmat = distmat.numpy()

        print('Computing CMC and mAP ...')
        cmc, mAP = metrics.eval_rank_market1501(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
        )

        return cmc, mAP

    def feature_extraction(self, loader):
        features_, pids_, camids_ = [], [], []
        with torch.no_grad():
            for _, data in enumerate(loader):
                imgs, pids, camids = self.parse_data_for_test(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
            
                features = self.model(imgs)
                features_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)
        features_ = torch.cat(features_, 0)
        pids_ = np.asarray(pids_)
        camids_ = np.asarray(camids_)
        return features_, pids_, camids_

    def parse_data_for_test(self, data):
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
        model.load_state_dict(torch.load(
            './pretrained_model.pth', map_location=device))

    return model

def print_result(cmc, mAP):
      print("Dataset statistics:")
      print("  ----------------------------------------")
      print("  distance  | rank1 | rank5 | mAP")
      print("  ----------------------------------------")
      print("  cosine    | %.2f  | %.2f  | %.2f " %(cmc[0], cmc[4], mAP))
      print("  ----------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = DataManager.add_model_specific_args(parser)
    parser.add_argument("--use_gpu", type=bool, default=False)

    args = parser.parse_args()

    dm = DataManager(args)

    model = models.OSNet(num_classes=dm.num_train_classes)
    model = load_model(model, args.use_gpu)

    tester = Tester(model, dm, args.use_gpu)

    cmc, mAP = tester.evaluate()
    print_result(cmc, mAP)
