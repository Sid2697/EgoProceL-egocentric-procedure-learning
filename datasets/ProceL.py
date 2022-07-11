"""
This file contains data loader for the ProcL dataset
"""

import os

import torch

import utils.logger as logging
from utils.utils import _extract_frames_h5py, _sample_frames_gen_labels_h5py


logger = logging.get_logger(__name__)


class ProceL(torch.utils.data.Dataset):
    """
    ProcL loader
    """
    def __init__(self, cfg, mode='all', transforms=None):
        self.cfg = cfg
        self.mode = mode
        self.transforms = transforms
        assert mode in ['train', 'test', 'val', 'all'], (
            'Wrong mode selected. Options are: train, val, test'
        )
        videos_dir = self.cfg.PROCEL.VIDEOS_DIR
        assert os.path.isdir(videos_dir), (
            f'Wrong videos path provided {videos_dir}!'
        )
        anns_dir = self.cfg.PROCEL.ANNS_DIR
        assert os.path.isdir(anns_dir), (
            f'Wrong annotations path provided {anns_dir}'
        )

        if self.mode == 'all':
            videos_path = os.path.join(videos_dir, self.cfg.PROCEL.CATEGORY)
            anns_path = os.path.join(anns_dir, self.cfg.PROCEL.CATEGORY)
        elif self.mode == 'train':
            raise NotImplementedError
        elif self.mode == 'test':
            raise NotImplementedError
        elif self.mode == 'val':
            raise NotImplementedError

        self.videos = [
            os.path.join(videos_path, file) for file in os.listdir(videos_path)
        ]
        self.annotations = [
            os.path.join(anns_path, file) for file in os.listdir(anns_path)
        ]

        self._construct_loader()

    def _construct_loader(self):
        """
        This method constructs the video and annotation data loader

        Returns:
            None
        """
        self.package = list()
        self.videos = sorted(
            self.videos, key=lambda a: a.split('/')[-1].split('mp4')[0].split('_')[-1]
        )
        for video in self.videos:
            video_name = video.split('/')[-1].split('mp4')[0]
            for annotation in self.annotations:
                if video_name in annotation:
                    self.package.append((video, annotation))

    def __len__(self):
        """
        Returns:
            (int): number of videos and annotations in the dataset
        """
        return len(self.package)

    def __getitem__(self, index):
        video_path, ann_path = self.package[index]
        h5_file_path = _extract_frames_h5py(
            video_path,
            self.cfg.PROCEL.FRAMES_PATH,
        )
        frames, labels = _sample_frames_gen_labels_h5py(
            self.cfg,
            h5_file_path,
            video_path,
            ann_path,
            transforms=self.transforms,
        )
        return frames, labels, video_path.split('/')[-1].split('mp4')[0]
