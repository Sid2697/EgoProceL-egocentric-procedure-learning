
import os
import pickle

import torch
import numpy as np

import utils.logger as logging
from datasets.transforms import get_transforms
from datasets.construct_loader import get_loader
from utils.parser import parse_args, load_config
from utils.subset_selection import SelfSupervisionSummarization
from utils.utils import get_category_metadata
from RepLearn.TCC.utils import (
    get_model,
    get_embds,
    gen_print_results,
    run_kmeans,
    graphcut_segmentation,
    random_segmentation,
)


logger = logging.get_logger(__name__)


def procedure_learning(cfg):
    # Fixing the seed to get the same results every time
    os.environ['PYTHONHASHSEED'] = str(cfg.TCC.RANDOM_STATE)
    np.random.seed(cfg.TCC.RANDOM_STATE)
    torch.manual_seed(cfg.TCC.RANDOM_STATE)

    # Getting metadata
    if cfg.ANNOTATION.DATASET_NAME == 'MECCANO':
        num_keysteps = 17
    elif cfg.ANNOTATION.DATASET_NAME == 'EPIC-Tents':
        num_keysteps = 12
    elif cfg.ANNOTATION.DATASET_NAME == 'pc_assembly':
        num_keysteps = 9
    elif cfg.ANNOTATION.DATASET_NAME == 'pc_disassembly':
        num_keysteps = 9
    else:
        category_metadata = get_category_metadata(cfg)
        num_keysteps = int(category_metadata['num_keysteps'])

    # Enable logging
    logging.setup_logging(cfg.LOG.DIR, cfg.LOG.LEVEL.lower(), cfg.LOG.BYPASS)

    # Loading the model
    model = get_model(cfg)
    try:
        model.load_state_dict(torch.load(cfg.TCC.MODEL_PATH)['state_dict'])
    except RuntimeError:
        # When the model is trained using data parallel class
        state_dict = torch.load(cfg.TCC.MODEL_PATH)['state_dict']
        new_state_dict = dict()
        for key, value in state_dict.items():
            new_state_dict[key.replace('module.', '')] = value
        model.load_state_dict(new_state_dict)
    model = model.to('cuda:1')
    model.eval()

    # Generating features
    gt = list()
    embeddings = list()
    all_frames = list()
    average_iou = list()
    average_recall = list()
    average_precision = list()

    if cfg.TCC.SUBSET_SELECTION:
        # Initialising subset selection
        subset_selection = SelfSupervisionSummarization(
            cfg.TCC.KMEANS_NUM_CLUSTERS,
            cfg.TCC.SUBSET_REPNUM,
            dim=cfg.TCC.EMBEDDING_SIZE
        )
        video_name_list = list()
        package = dict()

    # Load embeddings if saved previously, else create them
    embeddings_present = False
    model_name = cfg.TCC.MODEL_PATH.split('/')[-1].split('.pt')[0]
    embds_path = os.path.join(cfg.TCC.EMBDS_DIR, f'{model_name}_embds.pkl')
    if os.path.isfile(embds_path):
        embeddings_present = True
        print(f"Using embeddings from {embds_path}...")
        saved_embeddings = pickle.load(open(embds_path, 'rb'))
    else:
        print("Embeddings do not exist, creating new...")
        if not os.path.isdir(cfg.TCC.EMBDS_DIR):
            os.makedirs(cfg.TCC.EMBDS_DIR)
        to_save_embds = dict()

    data_loader = get_loader(cfg, mode='all', transforms=get_transforms(cfg))
    for sample in data_loader:
        frames, label, video_name = sample
        video_name = video_name[0]
        num_frames = frames.shape[1]
        if not embeddings_present:
            embds = get_embds(
                    model,
                    frames.squeeze().permute(0, 2, 3, 1),
                    num_frames,
                    cfg.TCC.EMBDS_BATCH,
                    cfg.TCC.NUM_CONTEXT_STEPS,
                    cfg.TCC.CONTEXT_STRIDE,
                    video_name
                )
            to_save_embds[video_name] = embds
        else:
            embds = saved_embeddings[video_name]
        if cfg.TCC.NORMALIZE_EMBDS:
            # Normalising the embeddings
            print('Normalising the embeddings...')
            embds = embds / np.linalg.norm(embds)
        assert len(embds) == num_frames

        # Subset selection
        if cfg.TCC.SUBSET_SELECTION:
            subset_selection.add_video(embds, video_name)
            video_name_list.append(video_name)
            package[video_name] = {
                'labels': label,
                'embds': embds,
                'frames': frames
            }
        else:
            # Evaluating at video level
            if cfg.TCC.GRAPH_CUT:
                kmeans_ind_preds = graphcut_segmentation(cfg, embds)
            elif cfg.TCC.RANDOM_RESULTS:
                kmeans_ind_preds = random_segmentation(cfg, embds)
            else:
                kmeans_ind_preds = run_kmeans(cfg, embds)
            recall, precision, iou, perm_gt, perm_pred = gen_print_results(
                cfg,
                label.squeeze(),
                kmeans_ind_preds,
                num_keysteps,
                video_name,
                return_assignments=True
            )
            average_iou.append(iou)
            average_recall.append(recall)
            average_precision.append(precision)

        embeddings.append(embds)
        gt.extend(label.squeeze().cpu().numpy())
        all_frames.extend(frames)

    embeddings_ = np.concatenate(embeddings, axis=0)
    assert len(gt) == embeddings_.shape[0]

    if cfg.TCC.SUBSET_SELECTION:
        # Subset selection overall preds
        overall_preds = subset_selection.forward()[1]

        # Subset selection for individual preds as it cannot be done in the
        # main loop
        for video_name_ in video_name_list:
            subset_preds = subset_selection.get_key_step_label(video_name_)
            labels = package[video_name_]['labels']
            embds_ = package[video_name_]['embds']
            frames_ = package[video_name_]['frames']
            recall, precision, iou = gen_print_results(
                cfg,
                labels.squeeze(),
                subset_preds.squeeze().detach().cpu().numpy(),
                num_keysteps,
                video_name_
            )
            average_iou.append(iou)
            average_precision.append(precision)
            average_recall.append(recall)
    else:
        if cfg.TCC.GRAPH_CUT:
            overall_preds = graphcut_segmentation(cfg, embeddings_)
        elif cfg.TCC.RANDOM_RESULTS:
            overall_preds = random_segmentation(cfg, embeddings_)
        else:
            overall_preds = run_kmeans(cfg, embeddings_)

    # Evaluate the entire thing
    import pdb; pdb.set_trace()
    gen_print_results(
        cfg,
        torch.from_numpy(np.array(gt)),
        overall_preds,
        num_keysteps,
        per_keystep=cfg.MISC.EVAL_PER_KEYSTEP,
    )

    # Saving the embeddings
    if not embeddings_present:
        print(f'Saving embeddings to {embds_path}...')
        pickle.dump(to_save_embds, open(embds_path, 'wb'))

    logger.critical(f'Average precision: {np.mean(average_precision)} '
            f'Average recall: {np.mean(average_recall)} '
            f'Average IoU: {np.mean(average_iou)}')


if __name__ == '__main__':
    procedure_learning(load_config(parse_args()))
