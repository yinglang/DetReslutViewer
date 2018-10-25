from view_utils import *
import matplotlib.pyplot as plt

# 1.load dataset
label_parser = CaltechLabelParser()
dataset = LstDataset_v2('dataset.lst', label_parser)

# 2. load detection result
result_dirs = ['detection_results/ssd', 'detection_results/FPN']
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
                                      show_text=False,
                                      class_names=None,
                                      use_real_line=[True, False, False, False],
                                      axes=axes, fig=fig)

    last_next_bt.update()
