import torch
import torch.nn.functional as F




def listNet(y_pred, y_true, weights, eps=0.1, padded_value_indicator=-1):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """


    y_pred = -1*y_pred.clone().float()
    y_true = -1*y_true.clone().float()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    weights_t= torch.from_numpy(weights).float()

    loss = torch.mean(-torch.sum(true_smax * preds_log, dim=1)*weights_t)
    return loss
