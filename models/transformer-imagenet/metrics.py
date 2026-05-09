import torch


def box_area(boxes):
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return widths * heights


def box_iou(pred_boxes, target_boxes):
    left_top = torch.maximum(pred_boxes[:, :2], target_boxes[:, :2])
    right_bottom = torch.minimum(pred_boxes[:, 2:], target_boxes[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    intersection = wh[:, 0] * wh[:, 1]

    union = box_area(pred_boxes) + box_area(target_boxes) - intersection
    return intersection / union.clamp(min=1e-6)


def generalized_box_iou(pred_boxes, target_boxes):
    left_top = torch.maximum(pred_boxes[:, :2], target_boxes[:, :2])
    right_bottom = torch.minimum(pred_boxes[:, 2:], target_boxes[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    intersection = wh[:, 0] * wh[:, 1]

    pred_area = box_area(pred_boxes)
    target_area = box_area(target_boxes)
    union = pred_area + target_area - intersection
    iou = intersection / union.clamp(min=1e-6)

    enclosing_left_top = torch.minimum(pred_boxes[:, :2], target_boxes[:, :2])
    enclosing_right_bottom = torch.maximum(pred_boxes[:, 2:], target_boxes[:, 2:])
    enclosing_wh = (enclosing_right_bottom - enclosing_left_top).clamp(min=0)
    enclosing_area = (enclosing_wh[:, 0] * enclosing_wh[:, 1]).clamp(min=1e-6)
    return iou - ((enclosing_area - union) / enclosing_area)


def generalized_box_iou_loss(pred_boxes, target_boxes):
    return 1.0 - generalized_box_iou(pred_boxes, target_boxes).mean()


def center_distance(pred_boxes, target_boxes):
    pred_center = (pred_boxes[:, :2] + pred_boxes[:, 2:]) * 0.5
    target_center = (target_boxes[:, :2] + target_boxes[:, 2:]) * 0.5
    return torch.linalg.vector_norm(pred_center - target_center, dim=1)


def classification_accuracy(logits, labels, topk=(1, 5)):
    max_k = min(max(topk), logits.shape[-1])
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(labels.reshape(1, -1))

    result = {}
    for k in topk:
        k = min(k, logits.shape[-1])
        result[f"top{k}"] = correct[:k].any(dim=0).float().mean()
    return result


def localization_metrics(outputs, targets, iou_threshold=0.5):
    iou = box_iou(outputs["box"], targets["box"])
    giou = generalized_box_iou(outputs["box"], targets["box"])
    center_error = center_distance(outputs["box"], targets["box"])
    accuracy = classification_accuracy(outputs["class_logits"], targets["label"])
    return {
        "top1": accuracy["top1"],
        "top5": accuracy["top5"],
        "mean_iou": iou.mean(),
        "mean_giou": giou.mean(),
        "center_error": center_error.mean(),
        "loc_at_iou": (iou >= iou_threshold).float().mean(),
    }
