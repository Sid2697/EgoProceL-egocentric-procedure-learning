
import utils.logger as logging

import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


logger = logging.get_logger(__name__)


def accuracy(assignment, labels):
    labels = labels.squeeze()
    assert len(assignment) == labels.shape[0]
    total = 0
    correct = 0
    for i in range(len(assignment)):
        total += 1
        if assignment[i] == labels[i]:
            correct += 1
    accuracy = correct/total
    return accuracy


def compute_align_MoF_UoI(
    keystep_pred,
    keystep_gt,
    n_keystep,
    M=None,
    per_keystep=False,
    return_assignments=False
):
    try:
        keystep_pred = torch.FloatTensor(keystep_pred)
    except:
        pass
    # breakpoint()
    # try:
    #     Z_pred = F.one_hot(
    #         keystep_pred.to(torch.int64),
    #         n_keystep
    #     ).float().cpu().numpy()
    # except RuntimeError:
    # if keystep_pred.min() < 0:
    #     # (29-10-2021): Needed for evaluating Kukleva's work when modelling the background
    #     keystep_pred += 1
    # Z_pred = F.one_hot(
    #     keystep_pred.to(torch.int64),
    #     M
    # ).float().cpu().numpy()
    if type(keystep_pred) == torch.Tensor:
        keystep_pred = keystep_pred.detach().cpu().numpy()
    Z_pred = torch.eye(M)[keystep_pred.astype(np.int32), :].float().cpu().numpy()
    # Z_gt = F.one_hot(
    #     keystep_gt.to(torch.int64),
    #     n_keystep
    # ).float().cpu().numpy()
    Z_gt = torch.eye(n_keystep)[keystep_gt, :].float().cpu().numpy()

    assert Z_pred.shape[0] == Z_gt.shape[0]
    T = Z_gt.shape[0]*1.0

    Dis = 1.0 - np.matmul(np.transpose(Z_gt), Z_pred)/T

    perm_gt, perm_pred = linear_sum_assignment(Dis)
    logger.critical('perm_gt {} perm_pred {}'.format(perm_gt, perm_pred))

    Z_pred_perm = Z_pred[:, perm_pred]
    Z_gt_perm = Z_gt[:, perm_gt]

    if per_keystep:
        list_MoF = []
        list_IoU = []
        list_precision = []
        step_wise_metrics = dict()
        for count, idx_k in enumerate(range(Z_gt_perm.shape[1])):
            pred_k = Z_pred_perm[:, idx_k]
            gt_k = Z_gt_perm[:, idx_k]

            intersect = np.multiply(pred_k, gt_k)
            union = np.clip((pred_k + gt_k).astype(np.float), 0, 1)

            n_intersect = np.sum(intersect)
            n_union = np.sum(union)
            n_predict = np.sum(pred_k)

            n_gt = np.sum(gt_k == 1)

            if n_gt != 0:
                MoF_k = n_intersect/n_gt
                IoU_k = n_intersect/n_union
                if n_predict == 0:
                    Prec_k = 0
                else:
                    Prec_k = n_intersect/n_predict
            else:
                MoF_k, IoU_k, Prec_k = [-1, -1, -1]
            list_MoF.append(MoF_k)
            list_IoU.append(IoU_k)
            list_precision.append(Prec_k)
            step_wise_metrics[count] = {
                "MoF": MoF_k,
                "IoU": IoU_k,
                "prec": Prec_k
            }

        arr_MoF = np.array(list_MoF)
        arr_IoU = np.array(list_IoU)
        arr_prec = np.array(list_precision)

        mask = arr_MoF != -1
        MoF = np.mean(arr_MoF[mask])
        IoU = np.mean(arr_IoU[mask])
        Precision = np.mean(arr_prec[mask])
        if return_assignments:
            return None, None, None, perm_gt, perm_pred
        else:
            return MoF, IoU, Precision, step_wise_metrics
    else:
        intersect = np.multiply(Z_pred_perm, Z_gt_perm)
        union = np.clip((Z_pred_perm + Z_gt_perm).astype(np.float), 0, 1)

        n_intersect = np.sum(intersect)
        n_union = np.sum(union)
        n_predict = np.sum(Z_pred_perm)

        n_gt = np.sum(Z_gt_perm)

        MoF = n_intersect/n_gt
        IoU = n_intersect/n_union
        Precision = n_intersect/n_predict
    if return_assignments:
        return None, None, None, perm_gt, perm_pred
    else:
        return MoF, IoU, Precision, None


def procl_eval(preds, labels, return_assignments=False):
    """
    This method takes in lists of predictions and labels, finds a mapping
    between them using the Hungarian algorithm and calculates the precision,
    recall and F1 scores.
    Original code source: github.com/Yuhan-Shen/VisualNarrationProceL-CVPR21

    Args:
        labels (ndarray): Numpy array of ground truth labels
        preds (ndarray): Numpy array of predictions
        return_assignment (bool): If true return Hungarian assignments

    Returns:
        precision (float): precision score
        recall (float): recall score
        F1 (float): harmonic mean of precision and recall
    """
    assert len(labels) == len(preds)
    # Adding 1 to take `0` label and prediction into account
    k_pred = int(preds.max()) + 1
    k_label = int(labels.max()) + 1
    # Generating the cost matrix. Here the labels and predictions with higher
    # overlap have higher numbers
    overlap = np.zeros([k_pred, k_label])
    for i in range(k_pred):
        for j in range(k_label):
            overlap[i, j] = np.sum((preds==i) * (labels==j))
    # Converting maximum overlap to minimum assignment cost and normalizing the
    # matrix
    # Assigning labels to predictions using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-overlap / preds.shape[0])
    # For the cases when the number of unique labels and number of unique
    # predictions are not equal, we will need to add additional term to make
    # the length of label assignment and pred assignment equal
    K = max(k_pred, k_label)
    bg_row_ind = np.concatenate(
        [row_ind, -np.ones(K+1-row_ind.shape[0], dtype=np.int32)]
    )
    bg_col_ind = np.concatenate(
        [col_ind, -np.ones(K+1-col_ind.shape[0], dtype=np.int32)]
    )
    print(bg_row_ind, bg_col_ind)
    acc = np.mean(bg_col_ind[preds]==bg_row_ind[labels])
    acc_steps = np.mean(
        bg_col_ind[preds[labels>=0]]==bg_row_ind[labels[labels>=0]]
    )

    results = []
    # Calculating precision and recall in a step-wise manner
    for i, p in enumerate(row_ind):
        # Cases where label = ground truth labels and label = preidctions
        # (after the hungarian assignment)
        correct = preds[labels==col_ind[i]] == p
        if correct.shape[0] == 0:
            num_correct = 0
        else:
            num_correct = np.sum(correct)
        num_label = np.sum(labels==col_ind[i])
        num_pred = np.sum(preds==p)
        results.append([num_correct, num_label, num_pred])

    # Taking into account the cases which we might have missed when they are
    # not matched using the Hungarian algorithm (due to this they are not in
    # the output of the algorithm and as in the previous loop we loop only
    # on the algorithm's output)
    for i in range(k_pred):
        if i not in row_ind:
            num_correct = 0
            num_label = 0
            num_pred = np.sum(preds==i)
            results.append([num_correct, num_label, num_pred])
    for j in range(k_label):
        if j not in col_ind:
            num_correct = 0
            num_label = np.sum(labels==j)
            num_pred = 0
            results.append([num_correct, num_label, num_pred])

    results = np.array(results)
    precision = np.sum(results[:, 0]) / (np.sum(results[:, 2]) + 1e-10)
    recall = np.sum(results[:, 0]) / (np.sum(results[:, 1]) + 1e-10)
    fscore = 2 * precision * recall / (precision + recall + 1e-10)
    if return_assignments:
        return row_ind, col_ind
    return [precision, recall, fscore, acc, acc_steps]


if __name__ == '__main__':
    label_list = [np.random.randint(0, 5, [i]) for i in np.random.randint(10, 15, 10)]
    lens = [label.shape[0] for label in label_list]
    pred_list = [np.random.randint(0, 4, [i]) for i in lens]
    preds = np.concatenate(pred_list)
    labels = np.concatenate(label_list)
    metric = procl_eval(preds, labels)
    print(metric)
    print(f'\n{compute_align_MoF_UoI(torch.tensor(np.hstack(pred_list)), torch.tensor(np.hstack(label_list)), 5, M=30)}')
    print(f'\n{compute_align_MoF_UoI(torch.tensor(np.hstack(pred_list)), torch.tensor(np.hstack(label_list)), 5, M=30, per_keystep=True)}')
    np_custom_gt = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 5])
    np_custom_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    custom_pred = torch.tensor(np_custom_pred)
    custom_gt = torch.tensor(np_custom_gt)
    print('Custom results...')
    print(f'\nCVPR: {procl_eval(np_custom_pred, np_custom_gt)}\n')
    print(f'\nECCV: {compute_align_MoF_UoI(custom_pred, custom_gt, 6, M=30)}\n')
    print(f'\nECCV: {compute_align_MoF_UoI(custom_pred, custom_gt, 6, M=30, per_keystep=True)}\n')
