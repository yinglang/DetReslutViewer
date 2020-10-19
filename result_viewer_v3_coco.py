from view_utils import *
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='view detection result')
    parser.add_argument('--lst-file', dest='lst_file', help='dataset list file, use to load dataset',
                        default='dataset.lst', type=str)
    parser.add_argument('--det-results', dest='det_results', help='detection results dir',
                        default='detection_results/FPN,detection_results/ssd', type=str)
    parser.add_argument('--score-thresholds', dest='thresholds', help="detection results score's threshold.",
                        default='0.25,0.25', type=str)
    parser.add_argument('--show-titles', dest='combinations_name', help="detection results score's threshold.",
                        default='FPN,ssd,FPN+ssd', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 1.load dataset
    label_parser = CaltechLabelParser()
    dataset = LstDataset_v2(args.lst_file, label_parser)

    # 2. load detection result
    result_dirs = args.det_results.split(',')
    result_datas = []
    result_datas_name = []
    for result_dir in result_dirs:
        result_datas.append(CaltechResultData(result_dir, False))
        # assert len(dataset) == len(result_datas[-1]), result_dir

    # 3. show images
    thresholds = [float(e) for e in args.thresholds.split(',')]
    combinations_name = args.combinations_name.split(',')
    
    var = {'index': 0, 'exit': False}
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    last_next_bt = LastNextButtonV3(fig, axes, var, interval=0.1)
    while not var['exit']:
        var['index'] = var['index'] % len(dataset)
        data, label, vid = dataset[var['index']]
        vid[-1] += 1
        image_name = CaltechPathParser.id2path(vid)
        # print(data.shape, label.shape, image_name, result_data[image_name])
        if len(label) == 0: label = None
        bboxes = [result_data[image_name] for result_data in result_datas]
        print(image_name)
        data = data.astype(np.float32)
        fig, axes = show_multi_det_result(data/255, bboxes, label, show_origin=True,
                                          thresholds=thresholds, #[0.845791, 0.936530],
                                          colors=['red', 'blue'],
                                          label_color='green',
                                          normalized_label=False,
                                          # figsize=(16, 10), MN=(2, 2),
                                          hwspace=(0.1, 0),
                                          show_combinations=[[0], [1], [0, 1]],
                                          combinations_name=['FPN', 'ATT', 'FPN+ATT'],
                                          show_text=False,
                                          class_names=None,
                                          use_real_line=[True, False, False, False],
                                          axes=axes, fig=fig)

        last_next_bt.wait_change()
