# Last Update: 19/12/2024
# Utility Functions useful for computing IoU metric during validation phase
# The wrapper function calculate_iou is what main program calls; match_bboxes and bbox_iou do the main work
# Source for bbox_iou() and match_bboxes: https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4 

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = xB - xA + 1
    interH = yB - yA + 1

    if interW <= 0 or interH <= 0:
        return -1.0

    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MIN_IOU = 0.0

    # Initialize IOU matrix
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i, :], bbox_pred[j, :])

    # Handle cases with more predictions than ground-truths
    if n_pred > n_true:
        diff = n_pred - n_true
        iou_matrix = np.concatenate((iou_matrix, np.full((diff, n_pred), MIN_IOU)), axis=0)

    # Handle cases with more ground-truths than predictions
    if n_true > n_pred:
        diff = n_true - n_pred
        iou_matrix = np.concatenate((iou_matrix, np.full((n_true, diff), MIN_IOU)), axis=1)

    # Use Hungarian algorithm to match the boxes
    idxs_true, idxs_pred = linear_sum_assignment(1 - iou_matrix)

    if not idxs_true.size or not idxs_pred.size:
        return np.array([]), np.array([]), np.array([]), np.array([])

    ious = iou_matrix[idxs_true, idxs_pred]

    # Filter valid matches with IOU > threshold
    sel_pred = idxs_pred < n_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = ious_actual > IOU_THRESH
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label

def calculate_iou(pred_boxes, true_boxes, IOU_THRESH=0.5):
    # Ensure pred_boxes and true_boxes are numpy arrays
    pred_boxes = np.array(pred_boxes)
    true_boxes = np.array(true_boxes)

    # Match the predicted and true bounding boxes
    idx_gt, idx_pred, ious, label = match_bboxes(true_boxes, pred_boxes, IOU_THRESH)

    if len(ious) == 0:
        return 0.0  # If no valid matches, return 0 IoU

    # Calculate the average IoU for valid matches
    avg_iou = np.mean(ious)
    return avg_iou
