
import os

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import utils.logger as logging


logger = logging.get_logger(__name__)


def get_category_metadata(cfg, metadata=None):
    """
    This function takes in metadata for all the categories and returns the data
    of the category being processed

    Args:
        metadata(list): information loaded from the metadata file

    Return:
        cat_data(dict): information about a particular category
    """
    if cfg.ANNOTATION.DATASET_NAME == 'EGTEA_GazeP':
        keysteps_dict = {
            'BaconAndEggs': 11,
            'Cheeseburger': 10,
            'ContinentalBreakfast': 10,
            'GreekSalad': 4,
            'PastaSalad': 8,
            'Pizza': 8,
            'TurkeySandwich': 6,
        }
        return {'num_keysteps': keysteps_dict[cfg.ANNOTATION.CATEGORY]}
    if cfg.ANNOTATION.DATASET_NAME == 'ProceL':
        keysteps_dict = {
            'make_pbj_sandwich': 10,
            'assemble_clarinet': 16,
            'change_tire': 18,
            'make_coffee': 12,
            'perform_cpr': 8,
            'jump_car': 14,
            'repot_plant': 10,
            'setup_chromecast': 12,
            'change_iphone_battery': 14,
            'make_smoke_salmon_sandwich': 9,
            'tie_tie': 14,
            'change_toilet_seat': 21,
        }
        return {'num_keysteps': keysteps_dict[cfg.PROCEL.CATEGORY]}
    if cfg.ANNOTATION.DATASET_NAME == 'CrossTask':
        keysteps_dict = {
            105253: 11,
            109972: 5,
            113766: 11,
            16815: 3,
            23521: 6,
            40567: 11,
            44047: 8,
            44789: 8,
            53193: 6,
            59684: 5,
            71781: 8,
            76400: 10,
            77721: 5,
            87706: 9,
            91515: 8,
            94276: 6,
            95603: 7,
        }
        return {'num_keysteps': keysteps_dict[cfg.CROSSTASK.CATEGORY]}
    if metadata is None:
        metadata = open(cfg.CMU_KITCHENS.METADATA_FILE, 'r').readlines()
    for count, data in enumerate(metadata):
        if cfg.ANNOTATION.CATEGORY in data.rstrip():
            useful_data = metadata[count:count+4]
            category_id = useful_data[1].rstrip()
            num_keysteps = useful_data[2].rstrip()
            keysteps = useful_data[3].rstrip().split(',')
            cat_data = {
                'category_id': category_id,
                'num_keysteps': num_keysteps,
                'keysteps': keysteps
            }
            return cat_data


def _extract_frames_h5py(video_path, frames_path):
    videocap = cv2.VideoCapture(video_path)
    video_name = video_path.split('/')[-1].split('.')[0]
    h5_file_path = os.path.join(frames_path, f'{video_name}.h5')
    if os.path.isfile(h5_file_path):
        return h5_file_path
    h5_file = h5py.File(h5_file_path, "w")
    frames = list()
    desired_shorter_side = 384
    while videocap.isOpened():
        success, frame = videocap.read()
        if not success:
            break
        else:
            original_height, original_width, _ = frame.shape
            if original_height < original_width:
                # Height is the shorter side
                new_height = desired_shorter_side
                new_width = np.round(
                    original_width*(desired_shorter_side/original_height)
                ).astype(np.int64)
            elif original_height > original_width:
                # Width is the shorter side
                new_width = desired_shorter_side
                new_height = np.round(
                    original_height*(desired_shorter_side/original_width)
                ).astype(np.int64)
            else:
                # Both are the same
                new_height = desired_shorter_side
                new_width = desired_shorter_side
            assert np.isclose(
                new_width/new_height,
                original_width/original_height,
                0.01
            ), f'{new_width/new_height}; {original_width/original_height}'
            frame = cv2.resize(
                frame,
                (256, 256),
                interpolation=cv2.INTER_AREA
            )
            frames.append(frame)
    videocap.release()
    frames_npy = np.array(frames)
    dataset = h5_file.create_dataset(
        "images",
        np.shape(frames_npy),
        h5py.h5t.STD_U8BE,
        data=frames_npy
    )
    h5_file.close()
    return h5_file_path


def _extract_video_frames(cfg, video_path, frames_path):
    """
    This method extract videos from a given set of videos and saves them
    to a directory.

    Args:
        video_path (str): path to the video to load

    Returns:
        video_folder (str): path to the folder where extracted frames are
            saved
    """
    # calculating video's fps
    videocap = cv2.VideoCapture(video_path)
    fps = int(videocap.get(cv2.CAP_PROP_FPS))

    video_name = video_path.split('/')[-1].split('.')[0]
    video_folder = os.path.join(frames_path, video_name)
    if os.path.isdir(video_folder):
        # Frames from video already saved
        if len(os.listdir(video_folder)) > 0:
            # if cfg.MISC.VERBOSE:
            #     logger.info(f'{video_folder} exists...')
            return video_folder
        else:
            pass
    else:
        # if cfg.MISC.VERBOSE:
        #     logger.info(f'Extracting frames for {video_folder}...')
        os.makedirs(video_folder)

    frame_count = 0
    save_path = os.path.join(video_folder, '{0:0>7}_{1}.jpg')
    while videocap.isOpened():
        success, frame = videocap.read()
        if not success:
            break
        else:
            frame_count += 1
            cv2.imwrite(
                save_path.format(
                    frame_count,
                    str(fps)
                ),
                frame
            )
    videocap.release()
    return video_folder


def gen_labels(fps, annotation_data, num_frames, dataset_name=None):
    """
    This method is used to generate labels for the test dataset.

    Args:
        fps (int): frame per second of the video
        annotation_data (list): list of procedure steps
        num_frames (int): number of frames in the video

    Returns:
        labels (ndarray): numpy array of labels with length equal to the
            number of frames
    """
    labels = np.ones(num_frames, dtype=int)*-1
    for step in annotation_data:
        if dataset_name == 'CrossTask':
            start_time = step[1]
            end_time = step[2]
            label = step[0]
        else:
            start_time = step[0]
            end_time = step[1]
            label = step[2].split()[0]
        start_frame = np.floor(start_time * fps)
        end_frame = np.floor(end_time * fps)
        for count in range(num_frames):
            if count >= start_frame and count <= end_frame:
                try:
                    labels[count] = int(label)
                except ValueError:
                    """
                    EGTEA annotations contains key-steps numbers as 1.
                    instead of 1
                    """
                    assert label[-1] == '.'
                    label = label.replace('.', '')
                    labels[count] = int(label)
    return labels


def _sample_frames_gen_labels(
    cfg,
    video_folder,
    annotation_path,
    transforms=None
):
    """
    This method is used for sampling frames from saved directory and
    generate corresponding hard or soft labels.

    Args:
        video_folder (str): path to the folder where extracted frames are
            saved
        annotation_path (str): path to the corresponding annotation file

    Returns:
        frames (ndarray): extracted frames
        labels (ndarray): generated labels
    """
    if cfg.MISC.VERBOSE:
        logger.debug(f'Sampling frames from {video_folder}')
    assert os.path.isdir(video_folder), "Frames not extracted"
    frames = os.listdir(video_folder)
    # Sorting the frames to preserve the temporal information
    frames = sorted(frames, key=lambda a: int(a.split('_')[0]))
    fps = int(frames[0].split('_')[-1].split('.')[0])
    sampling_fps = cfg.DATA_LOADER.SAMPLING_FPS
    video_duration = len(frames) / fps
    # Number of frames we want from every video
    num_frames_to_sample = int(sampling_fps * video_duration)
    candidate_frames, mask = _sample_clip(
        cfg,
        frames,
        num_frames_to_sample,
        video_folder,
        transforms=transforms
    )
    annotation_data = pd.read_csv(
        open(annotation_path, 'r'),
        header=None
    )
    labels_ = gen_labels(fps, annotation_data.values, len(frames))
    labels_mask = labels_ * mask
    labels = list()
    for label in labels_mask:
        if label != 0:
            if label == -1:
                labels.append(0)
            else:
                labels.append(label)
    return np.concatenate(candidate_frames), np.array(labels)


def _load_frame(cfg, frame_path, transforms=None):
    """
    This method is used to read a frame and do some pre-processing.

    Args:
        frame_path (str): Path to the frame

    Returns:
        frames (ndarray): Image as a numpy array
    """
    frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
    if transforms:
        frame = transforms(frame)
    else:
        frame = cv2.resize(frame, (
            cfg.DATA_LOADER.CROP_SIZE,
            cfg.DATA_LOADER.CROP_SIZE
        ))
    # For concatenating all the frames in a video
    frame = np.expand_dims(frame, axis=0).astype(np.float32)
    return frame


def _sample_clip(
    cfg,
    frames,
    no_frames_required,
    video_folder,
    transforms=None
):
    """
    This method is used to sample the frames in a way that we always have
    same number of output frames for videos with different lengths and
    different sampling rates.

    Args:
        frames (list): list of names of frames for the clip being processed
        no_frames_required (int): number of frames required to be extracted
            from the clip
        video_folder (str): path to the folder where all the frame
            from the clip are saved

    Returns:
        frames (list): list of loaded frames
        keyframe_candidates_list (list): list of distance between keyframe
            and other frames in terms of location
    """
    num_frames = len(frames)
    error_message = 'Can\'t sample more frames than there are in the video'
    assert num_frames >= no_frames_required, error_message
    lower_lim = np.floor(num_frames/no_frames_required)
    upper_lim = np.ceil(num_frames/no_frames_required)
    lower_frames = list()
    upper_frames = list()
    lower_mask = np.zeros(len(frames))
    upper_mask = np.zeros(len(frames))
    for count, frame in enumerate(frames):
        if (count + 1) % lower_lim == 0:
            frame_path = os.path.join(video_folder, frame)
            lower_frames.append(
                _load_frame(cfg, frame_path, transforms=transforms)
            )
            if len(lower_frames) <= no_frames_required:
                # Making sure we do not get excess 1s
                lower_mask[count] = 1
        if (count + 1) % upper_lim == 0:
            frame_path = os.path.join(video_folder, frame)
            upper_frames.append(
                _load_frame(cfg, frame_path, transforms=transforms)
            )
            if len(upper_frames) <= no_frames_required:
                # Making sure we do not get excess 1s
                upper_mask[count] = 1
    if len(upper_frames) < no_frames_required:
        return lower_frames[:no_frames_required], lower_mask
    else:
        return upper_frames[:no_frames_required], upper_mask


def _sample_frames_gen_labels_h5py(
    cfg,
    h5_file_path,
    video_path,
    annotation_path,
    transforms=None
):
    assert os.path.isfile(h5_file_path), "H5 file not saved."
    h5_file = h5py.File(h5_file_path, 'r')
    frames = h5_file['images']
    videocap = cv2.VideoCapture(video_path)
    fps = int(videocap.get(cv2.CAP_PROP_FPS))
    videocap.release()
    sampling_fps = fps/2
    video_duration = len(frames) / fps
    num_frames_to_sample = int(sampling_fps * video_duration)
    candidate_frames, mask = _sample_clip_h5py(
        cfg,
        frames,
        num_frames_to_sample,
        transforms=transforms
    )
    annotation_data = pd.read_csv(
        open(annotation_path, 'r'),
        header=None
    )
    labels_ = gen_labels(
        fps,
        annotation_data.values,
        len(frames),
        dataset_name=cfg.ANNOTATION.DATASET_NAME,
    )
    labels_mask = labels_ * mask
    labels = list()
    for label in labels_mask:
        if label != 0:
            if label == -1:
                labels.append(0)
            else:
                labels.append(label)
    return np.concatenate(candidate_frames), np.array(labels)


def _load_frame_h5py(cfg, frame, transforms=None):
    if transforms:
        frame_out = transforms(frame)
    else:
        frame_out = cv2.resize(frame, (
            cfg.DATA_LOADER.CROP_SIZE,
            cfg.DATA_LOADER.CROP_SIZE,
        ))
    frame_out = np.expand_dims(frame_out, axis=0).astype(np.float32)
    return frame_out


def _sample_clip_h5py(cfg, frames, num_frames_to_sample, transforms=None):
    num_frames = len(frames)
    error_message = 'Can\'t sample more frames than there are in the video'
    assert num_frames >= num_frames_to_sample, error_message
    lower_lim = np.floor(num_frames/num_frames_to_sample)
    upper_lim = np.ceil(num_frames/num_frames_to_sample)
    count = np.arange(1, frames.shape[0] + 1)
    lower_mask = (count % lower_lim) == 0
    lower_frames = frames[lower_mask, :]
    upper_mask = (count % upper_lim) == 0
    upper_frames = frames[upper_mask, :]
    if len(upper_frames) < num_frames_to_sample:
        return_lower_frames = list()
        for frame in tqdm(lower_frames, desc='Loading frames'):
            return_lower_frames.append(
                _load_frame_h5py(cfg, frame, transforms=transforms)
            )
        return (
            return_lower_frames,
            lower_mask * np.ones(lower_mask.shape, dtype=np.int8)
        )
    else:
        return_upper_frames = list()
        for frame in tqdm(upper_frames, desc='Loading frames:'):
            return_upper_frames.append(
                _load_frame_h5py(cfg, frame, transforms=transforms)
            )
        return (
            return_upper_frames,
            upper_mask * np.ones(upper_mask.shape, dtype=np.int8)
        )
