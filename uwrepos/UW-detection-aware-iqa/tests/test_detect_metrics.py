import numpy as np
import pytest

from uwiqa.detect.metrics import Dets, box_iou, dataset_map, image_scores, yolo_labels_to_gt

GT = yolo_labels_to_gt("0 0.25 0.25 0.2 0.2\n1 0.70 0.70 0.2 0.2\n0 0.70 0.25 0.1 0.1\n")


def pred(boxes, cls, conf):
    return Dets(np.array(boxes, float).reshape(-1, 4), np.array(cls), np.array(conf, float))


def test_labels_conversion_and_iou():
    assert np.allclose(GT.boxes[0], [0.15, 0.15, 0.35, 0.35])
    assert box_iou(GT.boxes, GT.boxes).diagonal() == pytest.approx(1.0)


def test_perfect_predictions():
    s = image_scores(pred(GT.boxes, GT.cls, [0.9, 0.8, 0.7]), GT)
    assert s["ap"] == pytest.approx(1.0) and s["ap50"] == pytest.approx(1.0) and s["f1"] == 1.0


def test_no_predictions():
    s = image_scores(pred([], [], []), GT)
    assert s["ap"] == 0.0 and s["f1"] == 0.0 and s["n_gt"] == 3


def test_localisation_error_hits_ap_not_ap50():
    b = GT.boxes.copy()
    b[:, [0, 2]] += 0.03  # IoU = 0.17/0.23 = 0.739 -> TP at thresholds .50-.70 only (5 of 10)
    s = image_scores(pred(b[:2], GT.cls[:2], [0.9, 0.8]), Dets(GT.boxes[:2], GT.cls[:2]))
    assert s["ap50"] == pytest.approx(1.0) and s["ap"] == pytest.approx(0.5)


def test_wrong_class_is_fp_for_f1():
    s = image_scores(pred(GT.boxes[:1], [5], [0.9]), Dets(GT.boxes[:1], GT.cls[:1]))
    assert s["ap"] == 0.0 and s["f1"] == 0.0


def test_low_conf_false_positive_ranked_after_tp_does_not_hurt_ap():
    p = pred(np.r_[GT.boxes, [[0.0, 0.8, 0.1, 0.9]]], np.r_[GT.cls, 0], [0.9, 0.8, 0.7, 0.1])
    assert image_scores(p, GT)["ap"] == pytest.approx(1.0)


def test_dataset_map_pools_images():
    g1, g2 = Dets(GT.boxes[:1], GT.cls[:1]), Dets(GT.boxes[1:2], GT.cls[1:2])
    perfect = dataset_map([pred(g1.boxes, g1.cls, [0.9]), pred(g2.boxes, g2.cls, [0.9])], [g1, g2])
    assert perfect["map"] == pytest.approx(1.0)
    half = dataset_map([pred(g1.boxes, g1.cls, [0.9]), pred([], [], [])], [g1, g2])
    assert half["per_class_map"] == {0: pytest.approx(1.0), 1: 0.0} and half["map"] == pytest.approx(0.5)
