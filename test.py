import torch

import numpy as np
import argparse

import models
import metrics
from data import DataManager

import matplotlib.pyplot as plt


class Tester(object):
    def __init__(self, model, dm, use_gpu):
        self.model = model
        self.dm = dm
        self.use_gpu = use_gpu
        self.calc_distmat()

    def compute_rank_market1501(self, max_rank=10):
        num_q, num_g = self.distmat.shape
        print('Computing CMC and mAP ...')
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query

        for q_idx in range(num_q):
            # compute cmc curve
            raw_cmc, _ = self.remove_duplication(q_idx)
            if not np.any(raw_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = raw_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        return cmc, mAP

    def visualize(self, q_index, limit=10):
        q_img, q_pid, _ = self.parse_data_for_test(self.dm.queryset[q_index])
        _, sorted_index = self.remove_duplication(q_index)
        print('Top 10 images are as follow:')
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, limit + 1, 1)
        ax.set_title('q %d\npid %d' % (q_index, q_pid))
        ax.axis('off')
        self.img_show(q_img)
        for i in range(limit):
            ax = plt.subplot(1, limit + 1, i+2)
            ax.axis('off')
            g_index = sorted_index[i]
            g_img, g_pid, _ = self.parse_data_for_test(
                self.dm.galleryset[g_index])
            self.img_show(g_img)
            if g_pid == q_pid:
                ax.set_title('g %d\npid %d' % (g_index, g_pid), color='green')
            else:
                ax.set_title('g %d\npid %d' % (g_index, g_pid), color='red')

    def calc_distmat(self):
        self.model.eval()
        q_features, self.q_pids, self.q_camids = self.feature_extraction(
            self.dm.test_loader().query,
        )
        g_features, self.g_pids, self.g_camids = self.feature_extraction(
            self.dm.test_loader().gallery,
        )
        print('Computing feature distance ...')
        distmat = metrics.cosine_distance(q_features, g_features)
        self.distmat = distmat.numpy()
        print('Sorting feature distance ...')
        self.sorted_indices = np.argsort(self.distmat, axis=1)
        self.matches = (self.g_pids[self.sorted_indices]
                        == self.q_pids[:, np.newaxis]).astype(np.int32)

    def feature_extraction(self, loader):
        features_, pids_, camids_ = [], [], []
        with torch.no_grad():
            for _, data in enumerate(loader):
                imgs, pids, camids = self.parse_data_for_test(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                features = self.model(imgs)
                features = features.cpu().clone()
                features_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)
        features_ = torch.cat(features_, 0)
        pids_ = np.asarray(pids_)
        camids_ = np.asarray(camids_)
        return features_, pids_, camids_

    def remove_duplication(self, q_index):
        q_pid = self.q_pids[q_index]
        q_camid = self.q_camids[q_index]

        order = self.sorted_indices[q_index]
        remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
        keep = np.invert(remove)

        cmc = self.matches[q_index][keep]
        sorted_index = order[keep]
        return cmc, sorted_index

    def img_show(self, tensor_image):
        plt.imshow(tensor_image.permute(1, 2, 0))

    def parse_data_for_test(self, data):
        imgs = data['img']
        pids = data['pid']
        camids = data['camid']
        return imgs, pids, camids


######################################################################################
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
    print("  distance  | rank1   | rank5   | mAP  ")
    print("  ----------------------------------------")
    print("  cosine    | %.2f  | %.2f  | %.2f "
          % (cmc[0]*100, cmc[4]*100, mAP*100))
    print("  ----------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = DataManager.add_model_specific_args(parser)
    parser.add_argument("--use_gpu", type=bool, default=False)
    parser.add_argument("--query_index", type=int, default=0)

    args = parser.parse_args()

    dm = DataManager(args)

    model = models.OSNet(num_classes=dm.num_train_classes)
    model = load_model(model, args.use_gpu)

    tester = Tester(model, dm, args.use_gpu)

    cmc, mAP = tester.compute_rank_market1501()
    print_result(cmc, mAP)
    
    tester.visualize(args.query_index)
