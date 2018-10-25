from view_utils import *
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='view detection result')
    parser.add_argument('--lst-file', dest='dataset list file', help='use to load dataset',
                        default='../../dataset/CaltechPedestrians/caltech/lst/test.lst', type=str)

# 1.load dataset
label_parser = CaltechLabelParser()
dataset = LstDataset_v2(lst_file, label_parser)

# 2. load detection result
result_dirs = ['/home/hui/dataset/CaltechPedestrians/origin/res/temp',
               '/home/hui/dataset/CaltechPedestrians/origin/res/caltech_1gpu_e2e_faster_rcnn_R-50-FPN_v1_08-13999-480x640-blur']
result_datas = []
for result_dir in result_dirs:
    result_datas.append(CaltechResultData(result_dir, False))
    # assert len(dataset) == len(result_datas[-1]), result_dir

# 3. show images
args = {'index': 0, 'exit': False}
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
last_next_bt = LastNextButtonV2(fig, axes, args)
while not args['exit']:
    args['index'] = args['index'] % len(dataset)
    data, label, vid = dataset[args['index']]
    vid[-1] += 1
    image_name = CaltechPathParser.id2path(vid)
    # print(data.shape, label.shape, image_name, result_data[image_name])
    if len(label) == 0: label = None
    bboxes = [result_data[image_name] for result_data in result_datas]
    print(image_name)
    fig, axes = show_multi_det_result(data/255, bboxes, label, show_origin=True,
                                      thresholds=[0.25, 0.25],
                                      colors=['red', 'blue'],
                                      label_color='green',
                                      normalized_label=False,
                                      # figsize=(16, 10), MN=(2, 2),
                                      hwspace=(0.1, 0),
                                      show_combinations=[[0], [1], [0, 1]],
                                      combinations_name=['temp', 'temp2', 'temp1+temp2'],
                                      use_real_line=[True, False, False, False],
                                      axes=axes, fig=fig)

    last_next_bt.update()
