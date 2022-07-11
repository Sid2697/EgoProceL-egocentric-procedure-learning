
import os
import csv
import time

import tqdm
import torch
import numpy as np
from torch import optim
import tensorflow as tf
from fcmeans import FCM
from maxflow.fastmin import aexpansion_grid
from sklearn.cluster import MiniBatchKMeans

import utils.logger as logging
from RepLearn.TCC.models import Embedder
from evaluate.metrics import compute_align_MoF_UoI


logger = logging.get_logger(__name__)


def get_model(cfg):
    model = Embedder(cfg.TCC.EMBEDDING_SIZE, cfg.TCC.NUM_CONTEXT_STEPS, cfg=cfg)
    return model


def get_optimizer(model, cfg):
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.TCC.LR,
        weight_decay=cfg.TCC.WEIGHT_DECAY
    )
    return optimizer


def save_checkpoint(state, logdir, filename='checkpoint.pt'):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    path = os.path.join(logdir, filename)
    torch.save(state, path)


def get_embds(
    model,
    video,
    seq_len,
    frames_per_batch,
    num_context_steps,
    context_stride,
    video_name
):
    embds = []
    num_batches = int(np.ceil(float(seq_len)/frames_per_batch))
    for i in tqdm.tqdm(range(num_batches), desc=video_name):
        steps = np.arange(i*frames_per_batch, (i+1)*frames_per_batch)
        steps = np.clip(steps, 0, seq_len-1)
        def get_context_steps(step):
            return tf.clip_by_value(
                tf.range(step - (num_context_steps - 1) * context_stride,
                        step + context_stride,
                        context_stride),
                        0, seq_len-1)
        steps_with_context = tf.reshape(
            tf.map_fn(get_context_steps, steps),
            [-1]
        )
        frames = video[steps_with_context.numpy()]
        frames = tf.cast(frames, tf.float32)
        frames = (frames/127.5)-1.0
        frames = tf.image.resize(frames, (168, 168))
        frames = tf.expand_dims(frames, 0)
        frames = torch.from_numpy(
            frames.numpy()
        ).permute(0, 1, 4, 2, 3)
        output = model(frames.to('cuda:1'))
        embds.extend(output.detach().cpu().numpy())
    embds = np.concatenate(embds, axis=0)
    embds = embds[:seq_len]
    assert len(embds) == seq_len
    return embds


def graphcut_segmentation(cfg, features, alpha=7, beta=0.2):
    """
    This method performs graph cut based temporal segmentation on the frames.
    Referred to as the PCM module in the paper.

    Args:
        cfg: configuration
        alpha (int): Weight for scaling the temporal weights.
            Used for debugging.
        beta (float): Weight for scaling the labels weights.
            Used for debugging.

    Returns:
        cmeans_ind_preds_graph (ndarray): A numpy array consisting of
        assingment of frames to labels. Its shape is equal to the number of
        frames sampled.
    """
    logger.critical('Running Fuzzy CMeans...')
    start = time.time()
    fuzzy_cmeans = FCM(n_clusters=cfg.TCC.KMEANS_NUM_CLUSTERS)
    fuzzy_cmeans.fit(features)
    logger.debug(
        f'Clustering done. Time taken {np.round(time.time() - start, 3)}'
        ' secs'
    )
    cluster_probs = np.array(fuzzy_cmeans.u)
    # Calulating cost of assigning a frame to a label
    cluster_probs_ = np.ones(cluster_probs.shape) - cluster_probs
    cluster_probs_ = cfg.REP_LEARN.GRAPH_CUT_BETA * cluster_probs_
    # Calculating the cost of assigning different labels to neighbors
    L = cfg.TCC.KMEANS_NUM_CLUSTERS
    levs = np.arange(0.5/L, 1, 1/L)
    V = cfg.REP_LEARN.GRAPH_CUT_ALPHA * np.abs(levs.reshape((-1, 1)) - \
        levs.reshape((1, -1)))
    # Performing alpha expansion to determine labels
    cmeans_ind_preds_graph = aexpansion_grid(cluster_probs_, V)
    if cfg.MISC.DEBUG:
        cluster_probs_ = np.ones(cluster_probs.shape) - cluster_probs
        cluster_probs_ = beta * cluster_probs_
        V = alpha * np.abs(levs.reshape((-1, 1)) - levs.reshape((1, -1)))
        cmeans_ind_preds_graph = aexpansion_grid(cluster_probs_, V)
    return cmeans_ind_preds_graph


def random_segmentation(cfg, features):
    logger.critical('Generating random predictions...')
    L = cfg.TCC.KMEANS_NUM_CLUSTERS
    random_predictions = np.random.randint(L, size=(features.shape[0],))
    return random_predictions


################ TCC Procedure Learning Utils #################################
def run_kmeans(cfg, features):
    logger.critical('Running KMeans...')
    start = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=cfg.TCC.KMEANS_NUM_CLUSTERS,
        init='k-means++',
        max_no_improvement=None
    ).fit(features)
    logger.debug(
        f'Clustering done. Time taken {np.round(time.time() - start, 3)} secs'
    )
    # Getting KMeans labels
    kmeans_preds = kmeans.labels_.copy()
    return kmeans_preds


def gen_print_results(
    cfg,
    gt,
    pred,
    num_keysteps,
    video_name=None,
    per_keystep=False,
    return_assignments=False
):
    recall, IoU, precision, step_wise_metrics = compute_align_MoF_UoI(
        pred,
        gt,
        num_keysteps + 1,
        M=cfg.TCC.KMEANS_NUM_CLUSTERS,
        per_keystep=per_keystep
    )
    if video_name:
        logger.critical(
            f"Results for {video_name}: Precision: {precision}, Recall: "
            f"{recall}, Step wise Results: {step_wise_metrics}")
        if return_assignments:
            _, _, _, perm_gt, perm_pred = compute_align_MoF_UoI(
                pred,
                gt,
                num_keysteps + 1, 
                M = cfg.TCC.KMEANS_NUM_CLUSTERS,
                per_keystep=per_keystep,
                return_assignments=return_assignments
            )
            return recall, precision, IoU, perm_gt, perm_pred
        return recall, precision, IoU
    else:
        logger.critical(
            f"Overall Results. Precision: {precision}, Recall: {recall}, IOU: "
            f"{IoU}, Step wise Results: {step_wise_metrics}")
        if len(cfg.LOG.SAVE_CUMULATIVE_RESULTS) > 0 and cfg.LOG.DIR is not None:
            # Saving the overall results to make the experimentation process
            # faster
            error = 'Please provide correct path to the csv file.'
            assert '.csv' in cfg.LOG.SAVE_CUMULATIVE_RESULTS, error
            experiment_name = cfg.LOG.DIR.split('/')[-1]
            to_write = [
                experiment_name,
                precision,
                recall,
                IoU,
                step_wise_metrics
            ]
            with open(cfg.LOG.SAVE_CUMULATIVE_RESULTS, 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(to_write)
        return None

################ TCC Procedure Learning Utils #################################


################ LAV Contrastive-IDM Loss Utils ###############################
def generate_unique_video_steps(embeddings, steps):
    """
    This method generates list of unique temporal frame IDs from a given set of
    repeated frame IDs.
    The data loader provided by the authors from TCC generates frame IDs in a
    repeatitive manner. But for LAV we need it for a single video.

    Args:
        embeddings (ndarray): embeddings of videos
        steps (ndarray): repeatitive steps for videos
    Return:
        unique_vid_steps (list): list of unique video steps
    """
    unique_vid_steps = list()
    num_videos = embeddings.shape[0]
    embeddings_size = embeddings.shape[1]
    assert num_videos * embeddings_size == steps.shape[0]
    for i in range(0, steps.shape[0], embeddings_size):
        unique_vid_steps.append(steps[i])
    assert len(unique_vid_steps) == num_videos
    return unique_vid_steps


def get_lav_weights(steps):
    w_dash = np.zeros((len(steps), len(steps)))
    for i_count, i in enumerate(steps):
        for j_count, j in enumerate(steps):
            w_dash[i_count, j_count] = (i - j)**2 + 1
    w = 1/w_dash
    return w, w_dash
