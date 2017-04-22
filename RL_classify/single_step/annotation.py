import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET
import keras

class annotation:
    def __init__(self, image_set, data_path, with_bbox = True):
        # training set
        self._image_set = image_set
        # data path
        self._data_path = data_path

        self._image_ext = '.jpg'
        self._split_mark = ','
        self._classes = self.get_classes()
        self._sample_number = 0

        self._class_to_ind = dict(zip(self._classes, xrange(len(self._classes))))

        self._image_index, self._image_name, self._class_name = self._load_image_set_index_name()

        self._image_to_class = dict(zip(self._image_name, self._class_name))

        if with_bbox:
            self._roidb = self.gt_roidb()
        else:
            self._roidb = self.gt_imagedb()

    def get_classes(self):
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                     self._image_set + '.txt')
        with open(image_set_file) as f:
            class_info = f.readline().strip()
            class_len = class_info.split(self._split_mark)[0]
            return class_info.split(self._split_mark)[1:]

    def image_path_from_name(self, image_name):
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  image_name)
        return image_path

    def _load_image_set_index_name(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        image_names = []
        image_indexs = []
        class_names = []
        with open(image_set_file) as f:
            for line in f.readlines()[1:]:
                seperate_line = line.strip().split(self._split_mark)
                if(len(seperate_line) < 2):
                    break
                image_name = seperate_line[0]
                class_name = seperate_line[1]
                image_index = image_name[0:len(image_name) - 4]
                image_names.append(image_name)
                image_indexs.append(image_index)
                class_names.append(class_name)
        return image_indexs, image_names, class_names

    def _load_image_annotation(self, index):
        image_name = self.image_path_from_name(index)
        image = cv2.imread(image_name)
        boxes = np.zeros((1, 4), dtype=np.uint16)
        boxes[0, 0] = 0
        boxes[0, 1] = 0
        boxes[0, 2] = image.shape[1] - 1
        boxes[0, 3] = image.shape[0] - 1

        gt_classes = np.zeros((1), dtype=np.int32)
        gt_classes[0] = self._class_to_ind[self._image_to_class[index]]
        self._sample_number += 1

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'image_name': image_name}

    def _load_bbox_annotation(self, index):
        """
        Load image and bounding boxes info from XML file.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        # filter none positive annotations
        act_objs = [obj for obj in objs
                         if obj.find('name').text.lower().strip()
                         in self._classes]
        objs = act_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            cls_name = obj.find('name').text.lower().strip()
            cls = self._class_to_ind[cls_name]

            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls

            self._sample_number += 1

        # find the file name
        image_name = tree.find('filename').text.strip()

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'image_name': image_name}

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        """
        gt_roidb = [self._load_bbox_annotation(index)
                    for index in self._image_index]
        return gt_roidb

    def gt_imagedb(self):
        gt_roidb = [self._load_image_annotation(index)
                    for index in self._image_name]
        return gt_roidb

    def visulization_annatation(self, index):
        roi_db = self._roidb[index]
        boxes = roi_db['boxes']
        image_name = roi_db['image_name']
        gt_classes = roi_db['gt_classes']
        image_path = self.image_path_from_name(image_name)
        image = cv2.imread(image_path)
        for ibbox in xrange(boxes.shape[0]):
            bbox = boxes[ibbox, :]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            cv2.putText(image, self._classes[gt_classes[ibbox]], (bbox[0], bbox[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 255), thickness=2)
        cv2.putText(image, image_name, (30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=3)
        cv2.imshow("image", image)
        cv2.waitKey(0)

    def normalize_name(self, i, name_length = 6):
        name = str(i).rjust(name_length, '0')
        return name

    def image_from_bbox(self, data_path):
        # generate folders format: Annotations, ImageSets/Main, JPEGImages
        annotations_path = os.path.join(data_path, "Annotations")
        if not os.path.exists(annotations_path):
            os.makedirs(annotations_path)

        main_path = os.path.join(data_path, "ImageSets", "Main")
        if not os.path.exists(main_path):
            os.makedirs(main_path)

        image_path = os.path.join(data_path, "JPEGImages")
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        # open or create label file
        label_file = os.path.join(main_path, self._image_set+'.txt')
        output = open(label_file, 'w')

        # write class info
        class_info = str(len(self._classes))
        for cls_name in self._classes:
            class_info += self._split_mark + cls_name
        output.write(class_info)

        save_image_name = 0
        for ix, roi_db in enumerate(self._roidb):
            # self.visulization_annatation(ix)
            boxes = roi_db['boxes']
            image_name = roi_db['image_name']
            gt_classes = roi_db['gt_classes']
            original_image_path = self.image_path_from_name(image_name)
            image = cv2.imread(original_image_path)
            for ibbox in xrange(boxes.shape[0]):
                bbox = boxes[ibbox, :]
                image_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                cls_name = self._classes[gt_classes[ibbox]]
                save_image_name += 1
                image_name = self.normalize_name(save_image_name) + self._image_ext
                image_full_path = os.path.join(image_path, image_name)
                output.write('\n' + image_name + self._split_mark + cls_name)
                cv2.imwrite(image_full_path, image_crop)

    def prepare_keras_data(self, target_size = 256, color_mode = 'rgb'):

        if color_mode == 'rgb':
            dataset = np.zeros((self._sample_number, target_size, target_size, 3))
        else:
            dataset = np.zeros((self._sample_number, target_size, target_size, 1))
        label = np.zeros((self._sample_number, 1), dtype=np.int32)

        sample_index = 0
        for ix, roi_db in enumerate(self._roidb):
            # self.visulization_annatation(ix)
            boxes = roi_db['boxes']
            image_name = roi_db['image_name']
            gt_classes = roi_db['gt_classes']
            original_image_path = self.image_path_from_name(image_name)
            image = cv2.imread(original_image_path)
            for ibbox in xrange(boxes.shape[0]):
                bbox = boxes[ibbox, :]
                image_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

                cls_index = gt_classes[ibbox]
                image_crop = cv2.resize(image_crop, (target_size, target_size))
                if color_mode == 'rgb':
                    if np.shape(image_crop)[-1] == 3:
                        image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
                    else:
                        image_crop = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2RGB)
                else:
                    if np.shape(image_crop)[-1] == 3:
                        image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
                        image_crop = image_crop.reshape((image_crop.shape[0], image_crop.shape[1], 1))

                dataset[sample_index, :, :, :] = image_crop
                label[sample_index, 0] = int(cls_index)

                sample_index += 1
        keras_label = keras.utils.to_categorical(label, len(self._classes))

        return (dataset, keras_label)

def seperate_train_val_data(x_data, y_data, ratio = 0.8, min_sample_num = 5):
    sample_num = y_data.shape[0]
    if sample_num < min_sample_num:
        return (None, None), (None, None)
    else:
        train_num = int(sample_num * ratio)
        train_data = x_data[:train_num, :, :, :]
        train_label = y_data[:train_num, :]
        val_data = x_data[train_num:, :, :, :]
        val_label = y_data[train_num:, :]
        return (train_data, train_label), (val_data, val_label)
