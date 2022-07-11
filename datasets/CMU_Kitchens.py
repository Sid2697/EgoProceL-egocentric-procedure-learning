"""
This file contains data loader for the CMU Kitchens dataset
"""

import os

import torch

from utils.utils import (
    get_category_metadata,
    _extract_frames_h5py,
    _sample_frames_gen_labels_h5py
)
import utils.logger as logging
from annotations.metadata.CMU_Kitchens.misc import view_mapping_inv


logger = logging.get_logger(__name__)


class CMU_Kitchens(torch.utils.data.Dataset):
    """
    CMU Kitchens loader
    """
    def __init__(self, cfg, mode='all', transforms=None):
        # Loading necessary paths and data
        self.cfg = cfg
        self.mode = mode
        self.transforms = transforms
        assert mode in ['train', 'val', 'test', 'all'], ('Wrong model selected'
        '. Options are: train, val, test, all')
        videos_dir = self.cfg.CMU_KITCHENS.VIDEOS_PATH
        annotations_dir = self.cfg.CMU_KITCHENS.ANNS_PATH
        assert os.path.isdir(videos_dir), 'Wrong videos path provided!'
        assert os.path.isdir(annotations_dir), ('Wrong annotations path '
                                                    'provided!')
        view_id = view_mapping_inv[self.cfg.CMU_KITCHENS.VIEW]
        self.cat_info = get_category_metadata(self.cfg)
        self.category_id = self.cat_info['category_id'] + '_' + view_id
        if self.cfg.MISC.VERBOSE:
            logger.debug('Processing category: {}'.format(self.category_id))

        if self.mode == 'all':
            videos_path = os.path.join(videos_dir, self.category_id)
            anns_path = os.path.join(annotations_dir, self.category_id)
        elif self.mode == 'train':
            raise NotImplementedError
        elif self.mode == 'val':
            raise NotImplementedError
        elif self.mode == 'test':
            videos_path = videos_dir
            anns_path = annotations_dir

        self.videos = [
            os.path.join(videos_path, file) for file in os.listdir(videos_path)
        ]
        self.annotations = [
            os.path.join(anns_path, file) for file in os.listdir(anns_path)
        ]
        # assert len(self.videos) == len(self.annotations), 'Trahimaam!'

        self._construct_loader()

    def _construct_loader(self):
        """
        This method constructs the video and annotation loader

        Returns:
            None
        """
        self.package = list()
        # Sorting the videos based on subject number
        self.videos = sorted(
            self.videos, key=lambda a: a.split('/')[-1].split('_')[0]
        )
        for video in self.videos:
            video_name = video.split('/')[-1].split('.')[0]
            for annotation in self.annotations:
                if video_name in annotation:
                    self.package.append((video, annotation))
                    break

    def __len__(self):
        """
        Returns:
            (int): number of videos and annotations in the dataset
        """
        return len(self.package)

    def __getitem__(self, index):
        video_path, annotation_path = self.package[index]
        h5_file_path = _extract_frames_h5py(
            video_path,
            self.cfg.CMU_KITCHENS.FRAMES_PATH
        )
        frames, labels = _sample_frames_gen_labels_h5py(
            self.cfg,
            h5_file_path,
            video_path,
            annotation_path,
            transforms=self.transforms
        )
        return frames, labels, video_path.split('/')[-1].split('.')[0]
