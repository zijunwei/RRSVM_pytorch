import numpy as np
from sklearn.metrics import average_precision_score
from Metrics_Helper import calc_pr_ovr_noref


def meanAP(predictions, labels):
    # y_true: array, shape = [n_samples] or [n_samples, n_classes]
    # y_score: array, shape = [n_samples] or [n_samples, n_classes]
    return average_precision_score(labels, predictions)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    _, target = target.topk(maxk, 1, True, True)
    pred = pred.t()
    target = target.t()
    correct = pred.eq(target)

    res = []
    for k in topk:


        correct_k = correct[:k].float().sum(0)
        correct_k[correct_k>1] = 1
        correct_k = correct_k.sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def calculate_mAP(y_pred, y_true):
    num_classes = y_true.shape[1]
    average_precisions = []

    for index in range(num_classes):
        pred = y_pred[:, index]
        label = y_true[:, index]

        sorted_indices = np.argsort(-pred)
        sorted_pred = pred[sorted_indices]
        sorted_label = label[sorted_indices]

        tp = (sorted_label == 1)
        fp = (sorted_label == 0)

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        npos = np.sum(sorted_label)

        recall = tp * 1.0 / npos

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp * 1.0 / np.maximum((tp + fp), np.finfo(np.float64).eps)

        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        average_precisions.append(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

    # print(average_precisions)
    mAP = np.mean(average_precisions)

    return mAP


def accuracy_orig(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def match_accuracy(output, target):
    """Computes the precision@k for the specified values of k"""

    batch_size = target.size(0)
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    # for s_element in range(batch_size):
    target_indices = np.argwhere(target == 1)
    n_samples = len(target_indices)
    indices_per_item = [None]*batch_size
    for s_idx in target_indices:
        if indices_per_item[s_idx[0]] is None:
            indices_per_item[s_idx[0]] = [s_idx[1]]
        else:
            indices_per_item[s_idx[0]].append(s_idx[1])

    n_correct = 0

    for s_sample_idx in range(batch_size):
        s_output = np.argsort(-output[s_sample_idx])[:len(indices_per_item[s_sample_idx])]
        for s_idx in s_output:
            if s_idx in indices_per_item[s_sample_idx]:
                n_correct += 1

    res = n_correct * 1. / n_samples
    return res



def mAPNips2017(all_logits, all_labels):
  num_classes = all_logits.shape[1]
  APs = []
  for cid in range(num_classes):
    this_logits = all_logits[:, cid]
    this_labels = (all_labels == cid).astype('float32')
    if np.sum(this_labels) == 0:
      print('No positive videos for class {}. Ignoring...'.format(cid))
      continue
    _, _, _, ap = calc_pr_ovr_noref(this_labels, this_logits)
    APs.append(ap)
  mAP = np.mean(APs)
  return mAP, APs

# def mAPNips2017(all_logits, all_labels):
#   num_classes = all_logits.shape[1]
#   APs = []
#   for cid in range(num_classes):
#     this_logits = all_logits[:, cid]
#     this_labels = all_labels[:, cid]
#     # this_labels = (all_labels == cid).astype('float32')
#     if np.sum(this_labels) == 0:
#       print('No positive videos for class {}. Ignoring...'.format(cid))
#       continue
#     _, _, _, ap = calc_pr_ovr_noref(this_labels, this_logits)
#     APs.append(ap)
#   mAP = np.mean(APs)
#   return mAP, APs

if __name__ == '__main__':
    y_true1 = np.array([0, 0, 1, 1])
    y_scores1 = np.array([0.1, 0.4, 0.35, 0.8])
    AP1 = meanAP(y_scores1, y_true1)

    y_true2 = np.array([0, 1, 0, 1])
    y_scores2 = np.array([0.1, 0.4, 0.35, 0.8])
    AP2 = meanAP(y_scores2, y_true2)

    y_true = np.vstack([y_true1, y_true2])
    y_scores = np.vstack([y_scores1, y_scores2])

    AP = meanAP(y_scores.transpose(), y_true.transpose())

    y_true = np.hstack([y_true1, y_true2])
    y_scores = np.hstack([y_scores1, y_scores2])
    APC = meanAP(y_scores, y_true)

    print "DEBUG"


