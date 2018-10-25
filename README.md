# DetReslutViewer
viewer for detection result view and compare

## prequires
> * python3
> * numpy
> * cv2
> * matplotlib

## run demo
### 1. generate dataset lst file
```sh
python3 lst_generator.py --dataset-root=./caltech_subset --lst-file dataset.lst
```
for more detail to generate your dataset lst file, you can change or use function [generate_list] in lst_generator.py

### 2. run result_viewer to view detect result
```sh
python3 result_viewer_v3.py --lst-file=./dataset.lst --det-results=./detection_results
```

## more
### viewer arugument
```python
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
```
the default argument like this, you can change argument of show_multi_det_result in result_viewer_v3.py, to get more complex function.
like 
> * set show_origin=False to not show origin image.
> * change thresholds to only view result box that score not less than threshold for every detector results.
> * change colors for every detection results
> * change label_color for gt box color
> * change use_real_line to decide use real line or dash line for every class gt.
> * change show_combinations to decide sub-plot show which detection results.
> * change combinations_name to change every sub-plot's title.

### define LabelParser for you annotations
if you annotation is not Caltech style, you can define you self, LabelParser class and change 
CaltechLabelParser to your's. you visit CaltechLabelParser's code.
```py
label_parser = CaltechLabelParser()
dataset = LstDataset_v2(args.lst_file, label_parser)
```

### define you dataset
if don't want to default dataset, you can define yourself dataset, to load image and label, and load your dataset,
invoke [show_multi_det_result] to show your result. and use LastNextButtonV2 to add key event listen.

## version diff
> * V1: implement key event show, switch image by close figure and re-open a new figure
> * V2: switch image by cla and re-draw, not close figure.
> * V3: add command argument parser.