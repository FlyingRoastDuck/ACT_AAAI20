from __future__ import print_function, absolute_import
import os.path as osp
import os
import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import read_json
from ..utils.serialization import write_json


class msmt17(Dataset):
    url = 'https://drive.google.com/file/d/1PduQX1OBuoXDh9JxybYBoDEcKhSx_Q8j/view?usp=sharing'
    md5 = 'ea5502ae9dd06c596ad866bd1db0280d'

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(msmt17, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        self.load()

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import re
        import hashlib
        import tarfile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'MSMT17_V1.tar.gz')
        if osp.isfile(fpath) and \
                hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))

        # Extract the file
        exdir = osp.join(raw_dir, 'MSMT17_V1')
        if not osp.isdir(exdir):
            print("Extracting tar file")
            with tarfile.open(fpath) as tar:
                tar.extractall(raw_dir)

        # MSMT17 files
        def register(typeName, pattern=re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)'), test_query=False):
            assert typeName.lower() in ['gallery', 'query', 'train', 'val']
            nameMap = {
                'gallery': 'test', 'query': 'test',
                'train': 'train', 'val': 'train'
            }
            with open(osp.join(exdir, 'list_{}.txt'.format(typeName.lower())), 'r') as f:
                fpaths = f.readlines()
            fpaths = [name.strip().split(' ')[0] for name in fpaths]
            fpaths = sorted([osp.join(exdir, nameMap[typeName.lower()], name) for name in fpaths])
            curData = []
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, _, cam = map(int, pattern.search(fname).groups())
                cam -= 1
                curData.append((fpath, pid, cam))
            return curData

        self.train = register('train')
        self.val = register('val')
        self.trainval = self.train + self.val
        self.gallery = register('gallery')
        self.query = register('query')

    ########################
    # Added
    def load(self, verbose=True):
        trainPids = [pid[1] for pid in self.train]
        valPids = [pid[1] for pid in self.val]
        trainvalPids = [pid[1] for pid in self.trainval]
        galleryPids = [pid[1] for pid in self.gallery]
        queryPids = [pid[1] for pid in self.query]
        self.num_train_ids = len(set(trainPids))
        self.num_val_ids = len(set(valPids))
        self.num_trainval_ids = len(set(trainvalPids))
        self.num_query_ids = len(set(queryPids))
        self.num_gallery_ids = len(set(galleryPids))
        ##########
        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}".format(self.num_train_ids, len(self.train)))
            print("  val      | {:5d} | {:8d}".format(self.num_val_ids, len(self.val)))
            print("  trainval | {:5d} | {:8d}".format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}".format(self.num_query_ids, len(self.query)))
            print("  gallery  | {:5d} | {:8d}".format(self.num_gallery_ids, len(self.gallery)))
    ########################
