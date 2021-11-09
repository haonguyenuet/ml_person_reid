from utils.tools import read_image

class ImageDataset(object):
    """
    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        mode (str): 'train', 'query' or 'gallery'.
    """

    def __init__(
        self,
        data,
        transforms=None,
    ):
        self.data = data
        self.transforms = transforms

    def __getitem__(self, index):
        img_path, pid, camid = self.data[index]
        img = read_image(img_path)
        if self.transforms is not None:
            img = self.transforms(img)
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
        }
        return item

    def __len__(self):
        return len(self.data)

    def get_num_pids(self):
        pids = set()
        for items in self.data:
            pid = items[1]
            pids.add(pid)
        return len(pids)

    def get_num_cams(self):
        cams = set()
        for items in self.data:
            camid = items[2]
            cams.add(camid)
        return len(cams)
