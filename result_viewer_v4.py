from view_utils import *
import matplotlib.pyplot as plt
import argparse
from pycocotools.coco import COCO
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


class CocoDataset(object):
    def __init__(self, data_dir, ann_file, det_file=None):
        self.data_dir = data_dir
        self.cocoGt = COCO(ann_file)
        self.imgId = sorted(self.cocoGt.getImgIds())
        if det_file is not None:
            self.cocoDt = self.cocoGt.loadRes(resFile)
        else:
            self.cocoDt = None

    def __getitem__(self, idx):
        i = self.imgId[idx]
        if i in self.cocoGt.imgToAnns:
            bbox = self.turn_bbox(self.cocoGt.imgToAnns[i])
        else:
            bbox = [[-1] * 7]
        if self.cocoDt is not None and i in self.cocoDt.imgToAnns:
            det_bbox = self.turn_bbox(self.cocoDt.imgToAnns[i])
        else:
            det_bbox = [[-1]*7]
        img = self.cocoGt.imgs[i]
        im_name = img['im_name']
        vset_name = im_name.split('_')[0]
        im_path = os.path.join(data_dir, vset_name, im_name)
        img = cv2.imread(im_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.cocoDt is None:
            return img, np.array(bbox), im_name
        else:
            return img, np.array(bbox), np.array(det_bbox), im_name

    def turn_bbox(self, annos):
        """
        :param annos:
        :return: [[x1, y1, x2, y2, cid, score/difficult, ignore]]
        """
        bbox = []
        for anno in annos:
            box = anno['bbox'] + [anno['category_id']] +\
                  [0 if 'score' not in anno else anno['score']] +\
                  [anno['ignore'] if 'ignore' in anno else 0]
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]
            bbox.append(box)
        return bbox

    def __len__(self):
        return len(self.imgId)

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
