from view_utils import *
import matplotlib.pyplot as plt
import argparse
plt.interactive(False)


# def parse_args():
#     parser = argparse.ArgumentParser(description='view detection result')
#     parser.add_argument('--lst-file', dest='lst_file', help='dataset list file, use to load dataset',
#                         default='dataset.lst', type=str)
#     parser.add_argument('--det-results', dest='det_results', help='detection results dir',
#                         default='detection_results/FPN,detection_results/ssd', type=str)
#     parser.add_argument('--score-thresholds', dest='thresholds', help="detection results score's threshold.",
#                         default='0.25,0.25', type=str)
#     parser.add_argument('--show-titles', dest='combinations_name', help="detection results score's threshold.",
#                         default='FPN,ssd,FPN+ssd', type=str)
#     return parser.parse_args()

if __name__ == '__main__':
    # args = parse_args()

    # 1.load dataset and load detection result
    annFile = '/home/hui/github/DetectionProject/notebook_mxnet/Cityscapes/evaluation/val_gt.json'
    resFile = './coco_citypersons_val_citypersons_1gpu_e2e_faster_rcnn_R-50-FPN_v1_60-69999_dt_sort.json'
    data_dir = '/home/hui/dataset/Cityscapes/origin_img/leftImg8bit_trainvaltest/leftImg8bit/val/'
    threshold = 0.99
    dataset = CocoDataset(data_dir, annFile, resFile)

    # 3. show images
    var = {'index': 0, 'exit': False}
    fig, axes = plt.subplots(1, 1, figsize=(16, 10))
    last_next_bt = LastNextButtonV3(fig, axes, var, interval=0.1)
    while not var['exit']:
        var['index'] = var['index'] % len(dataset)
        data, gt, det, im_name = dataset[var['index']]
        print(data.shape, gt.shape, det.shape, im_name)
        show_result(data / 255, gt, det, ax=axes, score_th=threshold)

        last_next_bt.wait_change()
