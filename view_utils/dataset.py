"""
    label parser, path parser and other parser for data prepare
"""
import numpy as np
import os
import cv2


class LstDataset(object):
    """A dataset for loading from lst files, line store (image_path, label_path) pair fomat like::
        /home/user/dataset/test1.jpg /home/user/dataset/test2.txt
        /home/user/dataset/help1.png /home/user/dataset/help2.txt
        ......
        /home/user/dataset2/help1.png /home/user/dataset2/help2.txt

    path have no constaint in format
    Parameters
    ----------
    lst_file : str
        Path to lst_file.
    label_parser : callable object, LabelParser object.
    flag: 1 for color, 0 for gray.
    transform:

    Attributes
    ----------
    classes : list
        List of class names. `classes[i]` is the name for the integer label `i`
    items : list of tuples
        List of all ndarrays in (imagepath, labelpath) pairs.
    """

    def __init__(self, lst_file, label_parser, flag=1, transform=None):
        self._flag = flag
        self._transform = transform
        self._label_dict = {}
        self._exts = ['.png', '.jpg', '.bmp']
        self.synsets = []
        self.items = []
        self._list_images(lst_file)
        self.label_parser = label_parser
        self.label_parser.classes  # invoke to vertify if label_parser have implement classes property or not.

    def _list_images(self, root):
        for line in open(root).readlines():
            imgpath, labelpath = line.strip().split(' ')
            ext = (os.path.splitext(imgpath)[1]).lower()
            if ext not in self._exts:
                print('image suffix', ext, 'is not valid, ignore it.', 'ext must be one of', self._exts)
                continue
            self.items.append((imgpath, labelpath))

    def __getitem__(self, idx):
        """
            return
                data: nd.array (h, w, c)
                label: np.array, generate by label_parser
        """
        data = cv2.imread(self.items[idx][0], self._flag)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        label = self.label_parser(open(self.items[idx][1]).readlines(), data, self.items[idx])
        if self._transform is not None:
            return self._transform(data, label)
        return data, np.array(label)

    def __len__(self):
        return len(self.items)

    @property
    def classes(self):
        return self.label_parser.classes


class LstDataset_v2(LstDataset):
    def __getitem__(self, idx):
        data, label = super(LstDataset_v2, self).__getitem__(idx)
        return data, label, CaltechPathParser.path2id(self.items[idx][0])

class LabelParser(object):
    """
    base class for Label parse, for LstDataset
    """

    @property
    def classes(self):
        """
            return class_name list, such as ['person', 'people', 'person?', 'person-fa']
        """
        raise NotImplementedError()

    def __call__(self, label_strs, image, image_label_path_pair, *args, **kwargs):
        """
            from annotation file content 'label_strs' and image to get label
            return list(list) or np.array, cannot be nd.array
        """
        raise NotImplementedError()


def valid_box_filter(box):
    if box[0] >= box[2] or box[1] >= box[3]:
        return False
    return True


class CaltechLabelParser(LabelParser):
    def __init__(self, filter=valid_box_filter):
        self.label_dict = {'person': 0, 'people': 1, 'person?': 2, 'person-fa': 3}
        self.class_names = [''] * len(self.label_dict)
        for key in self.label_dict:
            self.class_names[self.label_dict[key]] = key
        self.filter = filter

    @property
    def classes(self):
        return self.class_names

    def __call__(self, lines, image, path_pair, *args, **kwargs):
        """
        input line format
            % Each object struct has the following fields:
            %  lbl  - a string label describing object type (eg: 'pedestrian')
            %  bb   - [l t w h]: bb indicating predicted object extent
            %  occ  - 0/1 value indicating if bb is occluded
            %  bbv  - [l t w h]: bb indicating visible region (may be [0 0 0 0])
            %  ign  - 0/1 value indicating bb was marked as ignore
            %  ang  - [0-360] orientation of bb in degrees
        output line format:
            [xmin, ymin, xmax, ymax, cid, difficult]
        """
        labels = []
        for line in lines:
            if line.strip()[0] == '%': continue
            lbl, l, t, w, h, occ, vl, vt, vw, vh, ign, ang = line.split(' ')
            cid = self.label_dict[lbl]
            xmin, ymin = float(l), float(t)
            xmax, ymax = xmin + float(w), ymin + float(h)
            if self.filter([xmin, ymin, xmax, ymax]):
                difficult = 0
                labels.append([xmin, ymin, xmax, ymax, cid, difficult])
        return np.array(labels)


class CaltechPathParser(object):
    def path2id(self, *args, **kwargs):
        return CaltechPathParser.path2id(*args, **kwargs)

    def id2path(self, *args, **kwargs):
        return CaltechPathParser.id2path(*args, **kwargs)

    @staticmethod
    def path2id(imgpath):
        idx = imgpath.rfind('/')
        if idx == -1:
            idx = imgpath.rfind('\\')
        imgname = imgpath[idx + 1:] if idx != -1 else imgpath  # set01_V003_I00089.jpg
        subname = imgname.split('_')
        setid = int(subname[0][3:])
        vid = int(subname[1][1:])
        idx = subname[2].rfind('.')
        iid = int(subname[2][1:idx])
        return [setid, vid, iid]

    @staticmethod
    def id2path(ids, root_dir='', suffix='.jpg'):
        setid, vid, iid = ids
        setid, vid, iid = str(int(setid)).zfill(2), str(int(vid)).zfill(3), str(int(iid)).zfill(5)
        name = '_'.join(['set' + setid, 'V' + vid, 'I' + iid + suffix])
        return os.path.join(root_dir, name)

class CaltechResultData(object):
    """
    Atrribute:
    ----------
        _data: dict([image_name, result]), result is [[x1, y1, x2, y2, class_id, scores]...], for class_id set to 0.
    """
    def __init__(self, result_dir, lazy_load=True):
        self._data = {}
        self._result_dir = result_dir
        if not lazy_load:
            for set_name in os.listdir(result_dir):
                set_dir = os.path.join(result_dir, set_name)
                for video_file in os.listdir(set_dir):
                    video_name = video_file[:-4]
                    self._load_result_data(set_name, video_name)

    def __getitem__(self, item):
        if item not in self._data:
            setid, vid, _ = CaltechPathParser.path2id(item)
            set_name, video_name = 'set' + str(setid).zfill(2), 'V' + str(vid).zfill(3)
            self._load_result_data(set_name, video_name)
        if item not in self._data:
            return np.array([])
        return self._data[item]

    def _load_result_data(self, set_name, video_name):
        result_file = os.path.join(self._result_dir, set_name, video_name + '.txt')
        result_file = open(result_file)
        results = {}
        for line in result_file.readlines():
            image_id, x, y, w, h, score = [float(e) for e in line.split(',')]
            if image_id not in results:
                results[image_id] = []
            results[image_id].append((x, y, x+w, y+h, 0, score))
        result_file.close()
        for image_id in results:
            image_name = '_'.join([set_name, video_name, 'I'+str(int(image_id)).zfill(5)+'.jpg'])
            self._data[image_name] = np.array(results[image_id])

    def __len__(self):
        return len(self._data)
