import torch
from torch import nn
from .eva import dc, hd95, asd, obj_asd,hd,sensitivity,precision,specificity,jc

class DiceLoss(nn.Module):
    """
    define the dice loss
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 0.00001
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def metrics_ratio(seg, gt):
    """
    define the jaccard ratio
    :param seg: segmentation result
    :param gt: ground truth
    :return:
    """
    #seg = seg.data.cpu().numpy()
    #gt = gt.data.cpu().numpy()
    seg = seg.flatten()
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0

    gt = gt.flatten()
    sum = gt.sum()
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0

    tp = (seg * gt).sum()  # jiao
    fp = seg.sum() - tp
    fn = gt.sum() - tp
    tn = sum - fp - tp - fn

    dice = 2 * float(tp+0.000001) / float(gt.sum() + seg.sum() + 0.000001)
    jaccard = float(tp+0.000001) / float(fp + fn + tp + 0.000001)
    precision = float(tp+0.000001) / float(fp + tp + 0.000001)
    recall = float(tp+0.000001) / float(fn + tp + 0.000001)  # = sensitivity

    return dice, jaccard, precision, recall
'''
def metrics_ratio(seg, gt):
    """
    define the jaccard ratio
    :param seg: segmentation result
    :param gt: ground truth
    :return:
    """
    #seg = seg.data.cpu().numpy()
    #gt = gt.data.cpu().numpy()


    return dc(seg,gt), jc(seg,gt), precision(seg,gt), sensitivity(seg,gt)'''


def metrics(seg, gt):
    """
    define the jaccard ratio
    :param seg: segmentation result
    :param gt: ground truth
    :return:
    """
    #seg = seg.data.cpu().numpy()
    #gt = gt.data.cpu().numpy()


    return dc(seg,gt),hd95(seg,gt), jc(seg,gt), precision(seg,gt), sensitivity(seg,gt),specificity(seg,gt)