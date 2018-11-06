# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import warnings
from .bbox_utils import *


def box_to_dashed_rect(ax, box, color, linewidth=1):
    x1, y1, x2, y2 = box
    ax.plot([x1, x2], [y1, y1], '--', color=color, linewidth=linewidth)
    ax.plot([x1, x2], [y2, y2], '--', color=color, linewidth=linewidth)
    ax.plot([x1, x1], [y1, y2], '--', color=color, linewidth=linewidth)
    ax.plot([x2, x2], [y1, y2], '--', color=color, linewidth=linewidth)


def box_to_rect(box, color, linewidth=1):
    """convert an anchor box to a matplotlib rectangle"""
    return plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                  fill=False, edgecolor=color, linewidth=linewidth)


def draw_bbox(fig, bboxes, color=(0, 0, 0), linewidth=1, fontsize=5, normalized_label=True, wh=None,
              show_text=False, class_names=None, class_colors=None, use_real_line=None, threshold=None):
    """
        draw boxes on fig

    argumnet:
        bboxes: [[x1, y1, x2, y2, (cid), (score) ...]],
        color: box color, if class_colors not None, it will not use.
        normalized_label: if label xmin, xmax, ymin, ymax is normaled to 0~1, set it to True and wh must given, else set to False.
        wh: (image width, height) needed when normalized_label set to True
        show_text: if boxes have cid or (cid, score) dim, can set to True to visualize it.

        use_real_line: None means all box use real line to draw, or [boolean...] means whether use real line for per class label.
        class_names: class name for every class.
        class_colors: class gt box color for every class, if set, argument 'color' will not use
    """
    if len(bboxes) == 0: return
    if np.max(bboxes) <= 1.:
        if normalized_label == False: warnings.warn(
            "[draw_bbox]:the label boxes' max value less than 1.0, may be it is noramlized box," +
            "maybe you need set normalized_label==True and specified wh", UserWarning)
    else:
        if normalized_label == True: warnings.warn(
            "[draw_bbox]:the label boxes' max value bigger than 1.0, may be it isn't noramlized box," +
            "maybe you need set normalized_label==False.", UserWarning)

    if normalized_label:
        assert wh != None, "wh must be specified when normalized_label is True. maybe you need setnormalized_label=False "
        bboxes = inv_normalize_box(bboxes, wh[0], wh[1])

    if color is not None and class_colors is not None:
        warnings.warn("'class_colors' set, then 'color' will not use, please set it to None")

    for box in bboxes:
        # [x1, y1, x2, y2, (cid), (score) ...]
        if len(box) >= 5 and box[4] < 0: continue  # have cid or not
        if len(box) >= 6 and threshold is not None and box[5] < threshold: continue
        if len(box) >= 5 and class_colors is not None: color = class_colors[int(box[4])]
        if len(box) >= 5 and use_real_line is not None and not use_real_line[int(box[4])]:
            box_to_dashed_rect(fig, box[:4], color, linewidth)
        else:
            rect = box_to_rect(box[:4], color, linewidth)
            fig.add_patch(rect)
        if show_text:
            cid = int(box[4])
            if class_names is not None: cid = class_names[cid]
            text = str(cid)
            if len(box) >= 6: text += " {:.3f}".format(box[5])
            fig.text(box[0], box[1], text,
                     bbox=dict(facecolor=(1, 1, 1), alpha=0.5), fontsize=fontsize, color=(0, 0, 0))


def show_multi_det_result(image, outs, label=None, thresholds=None, colors=['red', 'blue', 'magenta', 'black'], label_color='green',
                          linewidth=1, fontsize=5, normalized_label=True, wh=None, MN=None, show_text=False,
                          figsize=(8, 4), show_combinations=None, combinations_name=None, hwspace=None, show_origin=False,
                          use_real_line=None, class_names=None, axes=None, fig=None):
    """
    :param image: np.array, shape=(h, w, 3)
    :param outs: [detect_result１, detect_result2,  ..... detect_result_N],
                detect_resulti is numpy array, [[x1,y1,x2,y2,(class_id,　(score))]...], shape=(N, K), K>=4
    :param label: [[x1,y1,x2,y2,class_id,(clssid)]...], shape=(N,K), K>=4
    :param colors: colors for every detection results
    :param label_color: label box 's color
    :param normalized_label: box is normalize [0, 1] or not.
    :param wh: image shape, if use normalized_label=True, must given
    :param MN: subplot layout, for MN=(2, 3), will get 2*3 subplot
    :param figsize:
    :param show_combinations: sub-plot's content, [[0], [1], [0, 1]], means 0-subplot plot detect_result0,
            1-subplot plot detect_result1, and 2-subplot plot detect_result1 and detect_result2.
    :param combinations_name: sub-plot's title
    :param hwspace: subplot's distance of height and width
    :param show_origin: whether show origin image as last sub-plot.
    :param show_text: whether show text on box.
    :param use_real_line: whether use real line or dash line for every class gt box.
    :param class_names: class name for every class.
    :return: fig, axes of subplots.
    """
    if show_combinations is None:
        show_combinations = [(i,) for i in range(len(outs))]
        if len(outs) > 1: show_combinations.append(tuple(range(len(outs))))

    if axes is None:
        if not show_origin:
            M, N = MN if MN is not None else (len(show_combinations), 1)
        else:
            M, N = MN if MN is not None else (len(show_combinations) + 1, 1)
        fig, axes = plt.subplots(M, N, figsize=figsize)
    else:
        if isinstance(axes, np.ndarray):
            M = axes.shape[0]
            N = axes.shape[1] if len(axes.shape) > 1 else 1
        else:
            M, N = 1, 1

    if hwspace is not None: assert fig is not None, 'fig must given when hwspace is not None and axes given.'

    for i in range(M*N):
        if len(show_combinations) == 1: ax = axes
        elif M > 1 and N > 1: ax = axes[i//N][i%N]
        else: ax = axes[i]

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if i >= len(show_combinations):
            if i == len(show_combinations) and show_origin:
                ax.imshow(image)
                ax.set_title('origin')
            else: ax.set_visible(False)
            continue

        ax.imshow(image)
        title = combinations_name[i] if combinations_name is not None else 'combination ' + str(i)
        ax.set_title(title)
        for show_boxes_id in show_combinations[i]:
            bboxes = outs[show_boxes_id]
            color = colors[show_boxes_id]
            threshold = thresholds[show_boxes_id] if thresholds is not None else None
            if bboxes is not None and len(bboxes) > 0:
                draw_bbox(ax, bboxes, color, linewidth, fontsize, normalized_label, wh, show_text, class_names,
                          None, use_real_line, threshold)

        if label is not None and len(label) > 0:
            draw_bbox(ax, label, label_color, linewidth, fontsize, normalized_label, wh, show_text, class_names,
                      None, use_real_line)

    if hwspace is not None:
        fig.subplots_adjust(hspace=hwspace[0], wspace=hwspace[1])
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    return fig, axes


def show_result(image, gts, det_boxes, color=None, score_th=0.2, overlap_th=0.5, ignore_iod_th=0.5,
                ax=None, match_type=0):
    """
    match strange:
        'tp': det box iou match non-ignore gt, all is 'tp'
        'fp': det box iou match non-ignore gt, all is 'fp'
        'fn': det box not iou match non-ignore gt, gt is 'fn'
        'ignore': ignore det, ignore gt, det box not iou match non-ignore gt but iod match ignore gt, all is 'ignore'

    split gts to non-ignore gts and ignore gts
    match det box with non-ignore gts, get 'tp', 'fn'(miss gt)
    match rest det box with ignore gt, to get 'ignore' det box
    """
    if color is None:
        color = {'fn': 'b', 'fp': 'r', 'tp': 'g'}
    det_boxes = det_boxes[np.all([det_boxes[:, 4] >= 0, det_boxes[:, 5] >= score_th], axis=0)]
    gts = gts[np.all([gts[:, 4] >= 0, gts[:, 5] == 0], axis=0)]
    ignore_gt_logical = (gts[:, 6] == 1)                                       # find out ignore gts.
    ignore_gts = gts[ignore_gt_logical][:, :5]
    non_ignore_gts = gts[np.logical_not(ignore_gt_logical)][:, :5]

    ax.imshow(image)
    if len(ignore_gts) > 0:
        max_class = int(ignore_gts[:, 4].max())+1
        draw_bbox(ax, ignore_gts, color=color['tp'], use_real_line=[False]*max_class, normalized_label=False)
    if len(non_ignore_gts) == 0 and len(det_boxes) == 0: return
    elif len(det_boxes) == 0:
        draw_bbox(ax, non_ignore_gts, color=color['fn'], normalized_label=False)
        return
    elif len(non_ignore_gts) == 0:
        det_box_type = np.array(['fp'] * len(det_boxes))
    else:   #len(non_ignore_gts) > 0 and len(det_boxes) > 0:
        gt_box_type = np.array(['fn'] * len(non_ignore_gts))
        det_box_type = np.array(['fp'] * len(det_boxes))
        det_boxes = np.array(sorted(det_boxes, key=lambda x: -x[5]))            # sort by score, desc
        from mxnet import nd
        ious = nd.contrib.box_iou(nd.array(non_ignore_gts[:, :4]), nd.array(det_boxes[:, :4])).asnumpy()
        gt_indexs = ious.argmax(axis=0)
        # gt_indexs[ious.max(axis=0) < overlap_th] = -1
        if match_type == 0:    # 1gt vs 1 pred, max score pred, first gt.
            gt_select = np.array([False] * (len(non_ignore_gts)))
            for i, gt_index in enumerate(gt_indexs):
                if ious[gt_index, i] >= overlap_th:                                    # IOU >= overlap_th, matched
                    if not gt_select[gt_index]:
                        gt_select[gt_index] = True
                        det_box_type[i] = 'tp'
                        gt_box_type[gt_index] = 'tp'
        elif match_type == 1:   # 1 gt vs n pred
            gt_indexs[ious.max(axis=0) < overlap_th] = -1
            gt_box_type[gt_indexs] = 'tp'
            det_box_type[gt_indexs >= 0] = 'tp'
        draw_bbox(ax, non_ignore_gts[gt_box_type == 'fn'], color=color['fn'], normalized_label=False)

    rest_det_logical = det_box_type != 'tp'
    rest_det_boxes = det_boxes[rest_det_logical]
    if len(rest_det_boxes) > 0 and len(ignore_gts) > 0:
        rest_det_type = det_box_type[rest_det_logical]
        iods = bbox_iod(rest_det_boxes, ignore_gts).max(axis=1)
        print(np.sum(det_box_type == 'fp'), iods)
        rest_det_type[iods > ignore_iod_th] = 'ignore'
        det_box_type[rest_det_logical] = rest_det_type

    #draw_bbox(ax, gts[gt_box_type == 'tp'], color=color['tp'], use_real_line=[False], normalized_label=False)
    draw_bbox(ax, det_boxes[det_box_type == 'fp'], color=color['fp'], normalized_label=False)
    draw_bbox(ax, det_boxes[det_box_type == 'tp'], color=color['tp'], normalized_label=False)
