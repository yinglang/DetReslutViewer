import MPUtils as mpu
import os
import argparse


def write_line(anno_path, img_path, name):
    return img_path + " " + anno_path + '\n'


def get_image_path_name(anno_name, image_dir, image_suffixes):
    idx = anno_name.rfind('.')
    if idx == -1: idx = len(anno_name)  # if hvae no '.', mean it have no suffix.
    name = anno_name[:idx]
    for suffix in image_suffixes:
        imgpath = os.path.join(image_dir, name + suffix)
        if os.path.exists(imgpath):
            return os.path.abspath(imgpath), name
    raise ValueError('image name (' + image_dir + '/' + name + ') with suffix in'
                     + str(image_suffixes) + 'not exists, but annotation file exsit,' +
                     'please check image_suffixes or image_name')


def generate_list(dataset_root, lst_file, filter=None, write_line=write_line, image_suffixes=['.jpg'],
                  sub_dir={'annotations': 'annotations', 'images': 'images'}):
    """
        traversal all annotation file in (dataset_root + "/" + sub_dir['annotations']),
        filte them use given (filter),
        write the result of (write_line) to (lst_file) line by line.

        param:
            dataset_root: str, dataset root dir path, such as "caltech/train".
            lst_file: str, generated lst file path, such as "caltech/lst/train.lst"
            filter: function, return True will keep, False will ignore.
                    def filter(anno_path, image_path):
                        # anno_path: abs path of annotation file
                        # image_path: abs path of image file
                        # they are a path pair, such as filter('/home/user/xx.txt', '/home/user/xx.jpg')
                        return len(open(anno_path).readlines()) > 0 # not empty filter
            write_line: function, returned result will be writed to lst file as a line.
                    def write_line(anno_path, img_path, name):
                        # anno_path, img_path same as filter argument
                        # name is no suffix image name and annotation name
                        # such as write_line('/home/user/xx.txt', '/home/user/xx.jpg', 'xx')
            image_suffixes: list(str), image suffix must be one of them.
            sub_dir: dict('str':'str'), default is {'annotations':'annotations', 'images':'images'}
                    such as data dir structure
                    ----dataset_root
                    --------labels (all annotatons in here)
                    --------imageset (all images here)
                    then sub_dir={'annotations':'labels', 'images':'imageset'}
    """
    image_dir = dataset_root + '/' + sub_dir['images'] + "/"
    anno_dir = dataset_root + '/' + sub_dir['annotations'] + '/'

    list_f = open(lst_file, 'w')
    for txt in sorted(os.listdir(anno_dir)):
        anno_path = os.path.abspath(os.path.join(anno_dir, txt))
        img_path, name = get_image_path_name(txt, image_dir, image_suffixes)
        if filter is None or filter(anno_path, img_path):  # passed filter
            list_f.write(write_line(anno_path, img_path, name))
    list_f.close()


if __name__ == '__main__':
    print('ok')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', dest='dataset_root', help='pascal voc style dataset root dir, use to generate lst file',
                        default='caltech_subset', type=str)
    parser.add_argument('--lst-file', dest='lst_file', help='generate lst file path', default='./dataset.lst', type=str)
    args = parser.parse_args()
    print(args)
    generate_list(dataset_root=args.dataset_root, lst_file=args.lst_file)
