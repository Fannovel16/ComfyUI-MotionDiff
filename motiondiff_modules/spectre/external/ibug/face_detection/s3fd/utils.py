import torch
import numpy as np
from itertools import product


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep, 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def nms_np(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object, using numpy (for speed).
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    if scores.size(0) == 0:
        return [], 0
    else:
        areas = torch.mul(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]).cpu().numpy()
        x1, y1 = boxes[:, 0].cpu().numpy(), boxes[:, 1].cpu().numpy()
        x2, y2 = boxes[:, 2].cpu().numpy(), boxes[:, 3].cpu().numpy()
        scores = scores.cpu().numpy()
        order = scores.argsort()[: -top_k - 1: -1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

            w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
            ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

            inds = np.where(ovr <= overlap)[0]
            order = order[inds + 1]

        return keep, len(keep)


class Detect(object):

    def __init__(self, config):

        self.config = config

    def __call__(self, loc_data, conf_data, prior_data):

        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(num, num_priors, self.config.num_classes).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors, 4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4), batch_priors, self.config.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, self.config.num_classes, self.config.top_k, 5)

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.config.num_classes):
                c_mask = conf_scores[cl].gt(self.config.conf_thresh)
                scores = conf_scores[cl][c_mask]

                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                if self.config.use_nms_np:
                    ids, count = nms_np(boxes_, scores, self.config.nms_thresh, self.config.nms_top_k)
                else:
                    ids, count = nms(boxes_, scores, self.config.nms_thresh, self.config.nms_top_k)
                count = count if count < self.config.top_k else self.config.top_k

                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes_[ids[:count]]), 1)

        return output


class PriorBox(object):

    def __init__(self, input_size, feature_maps, config):

        self.imh = input_size[0]
        self.imw = input_size[1]
        self.feature_maps = feature_maps

        self.config = config

    def forward(self):
        mean = []
        for k, fmap in enumerate(self.feature_maps):
            feath = fmap[0]
            featw = fmap[1]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.config.prior_steps[k]
                f_kh = self.imh / self.config.prior_steps[k]

                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                s_kw = self.config.prior_min_sizes[k] / self.imw
                s_kh = self.config.prior_min_sizes[k] / self.imh

                mean += [cx, cy, s_kw, s_kh]

        output = torch.FloatTensor(mean).view(-1, 4)

        if self.config.prior_clip:
            output.clamp_(max=1, min=0)

        return output
