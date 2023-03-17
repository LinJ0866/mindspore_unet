import os
import numpy as np
import cv2
from PIL import Image, ImageSequence

from mindspore.mindrecord import FileWriter
import mindspore.dataset as de
import mindspore.ops as ops
import mindspore.ops.operations as F
import mindspore.common.dtype as mstype
from mindspore import Tensor

from config import cfg

class SegDataset:
    def __init__(self, dataset_name, mode, num_parallel_workers=4, num_reader = 1, shard_id=None, shard_num=None):
        self.mindrecord_save = os.path.join(self.mindrecord_save, mode)
        if dataset_name == 'VOC2012':
            self.dataset_dir = 'datasets/VOC2012'
            self.dataset_img_dir = os.path.join(self.dataset_dir, 'JPEGImages')
            self.dataset_ano_dir = os.path.join(self.dataset_dir,'SegmentationClass')
            self.dataset_ano_gray_dir = os.path.join(self.dataset_dir,'SegmentationClassGray')
            self.mindrecord_save =  os.path.join(self.dataset_dir,'VOC_mindrecord')
            if mode != 'test':
                if not(os.path.exists(self.dataset_ano_gray_dir)):
                    self.convert_to_gray()
                self.dataset_ano_dir = self.dataset_ano_gray_dir

            if mode == 'train' or mode == 'val':
                self.batch_size = cfg.train_batch_size
                self.repeat = cfg.train_repeat
                self.min_scale = cfg.train_min_scale
                self.max_scale = cfg.train_max_scale
                self.image_mean = cfg.train_image_mean
                self.image_std = cfg.train_image_std
                self.crop_size = cfg.train_crop_size
                if mode == 'train':
                    self.dataset_lst = os.path.join(self.dataset_dir,'ImageSets/Segmentation/train.txt')
                    self.mindrecord_save = os.path.join(self.mindrecord_save, 'train')
                else:
                    self.dataset_lst = os.path.join(self.dataset_dir,'ImageSets/Segmentation/val.txt')
                    self.mindrecord_save = os.path.join(self.mindrecord_save,'eval')

            else:
                self.dataset_lst = os.path.join(self.dataset_dir,'ImageSets/Segmentation/val.txt')
                self.batch_size = cfg.test_batch_size
                self.repeat = cfg.test_repeat
                self.min_scale = cfg.test_min_scale
                self.max_scale = cfg.test_max_scale
                self.image_mean = cfg.test_image_mean
                self.image_std = cfg.test_image_std
                self.crop_size = cfg.test_crop_size
        elif dataset_name == 'cell':
            self.dataset_dir = 'datasets/cell'
            self.dataset_train_img = os.path.join(self.dataset_dir, 'train-volume.tif')
            self.dataset_train_anno = os.path.join(self.dataset_dir, 'train-labels.tif')

        self.num_parallel_workers = num_parallel_workers
        self.num_readers = num_reader
        self.shard_id = shard_id
        self.shard_num = shard_num
        self.mode = mode

        if mode == 'train' or mode == 'val':
            self.batch_size = cfg.train_batch_size
            self.repeat = cfg.train_repeat
            self.min_scale = cfg.train_min_scale
            self.max_scale = cfg.train_max_scale
            self.image_mean = cfg.train_image_mean
            self.image_std = cfg.train_image_std
            self.crop_size = cfg.train_crop_size
        else:
            self.dataset_lst = os.path.join(self.dataset_dir,'ImageSets/Segmentation/val.txt')
            self.batch_size = cfg.test_batch_size
            self.repeat = cfg.test_repeat
            self.min_scale = cfg.test_min_scale
            self.max_scale = cfg.test_max_scale
            self.image_mean = cfg.test_image_mean
            self.image_std = cfg.test_image_std
            self.crop_size = cfg.test_crop_size

        
    
    def _load_multipage_tiff(self, path):
        """Load tiff images containing many images in the channel dimension"""
        return np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(path))])

    def preprocess_(self, image, label):
        image_out = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        label_out = cv2.imdecode(np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        
        sc = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(sc * image_out.shape[0]), int(sc * image_out.shape[1])
        image_out = cv2.resize(image_out, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        label_out = cv2.resize(label_out, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        image_out = (image_out - self.image_mean) / self.image_std
        h_, w_ = max(new_h, self.crop_size), max(new_w, self.crop_size)
        pad_h, pad_w = h_ - new_h, w_ - new_w
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label_out = cv2.copyMakeBorder(label_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=cfg.ignore_label)
        offset_h = np.random.randint(0, h_ - self.crop_size + 1)
        offset_w = np.random.randint(0, w_ - self.crop_size + 1)
        image_out = image_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size, :]
        label_out = label_out[offset_h: offset_h + self.crop_size, offset_w: offset_w+self.crop_size]

        if np.random.uniform(0.0, 1.0) > 0.5:
            image_out = image_out[:, ::-1, :]
            label_out = label_out[:, ::-1]

        image_out = image_out.transpose((2, 0, 1))
        # image_out = image_out.copy()
        # label_out = label_out.copy()
        
        # depth, on_value, off_value = 21, Tensor(1.0, mstype.float32), Tensor(0.0, mstype.float32)
        # label_out = F.Cast()(label_out, mstype.int32)
        # label_out = ops.OneHot()(label_out, depth, on_value, off_value)
        # image_out.astype(mstype.float32)
        # image_out = F.Cast()(image_out, mstype.float32)
        # label_out = F.Cast()(label_out, mstype.float32)
        return image_out, label_out

    def get_mindrecord_dataset(self, num_shards=1, shuffle=True):
        if not(os.path.exists(self.mindrecord_save)):        
            print('creating mindrecord dataset...')
            with open(self.dataset_lst) as f:
                lines = f.readlines()
            if shuffle:
                np.random.shuffle(lines)

            os.makedirs(self.mindrecord_save)
            self.mindrecord_save = os.path.join(self.mindrecord_save,'VOC_mindrecord')
            print('number of samples:', len(lines))
            seg_schema = {"file_name": {"type": "string"}, "label": {"type": "bytes"}, "data": {"type": "bytes"}}
            writer = FileWriter(file_name=self.mindrecord_save, shard_num=num_shards)
            writer.add_schema(seg_schema, "seg_schema")

            datas = []
            cnt = 0
            for l in lines:
                id_ = l.strip()
                img_path = os.path.join(self.dataset_img_dir, id_ + '.jpg')
                label_path = os.path.join(self.dataset_ano_dir, id_ + '.png')
                
                sample_ = {"file_name": img_path.split('/')[-1]}
                with open(img_path, 'rb') as f:
                    sample_['data'] = f.read()
                with open(label_path, 'rb') as f:
                    sample_['label'] = f.read()
                datas.append(sample_)
                cnt += 1
                if cnt % 1000 == 0:
                    writer.write_raw_data(datas)
                    print('number of samples written:', cnt)
                    datas = []
                
            if datas:
                writer.write_raw_data(datas)
            writer.commit()
            print('number of samples written:', cnt)
            print('Create Mindrecord Done')
        

        self.mindrecord_save = os.path.join(self.mindrecord_save,'VOC_mindrecord')

        data_set = de.MindDataset(dataset_files=self.mindrecord_save, columns_list=["data", "label"],
                                shuffle=True, num_parallel_workers=self.num_readers,
                                num_shards=self.shard_num, shard_id=self.shard_id)
        
        transforms_list = self.preprocess_
        data_set = data_set.map(operations=transforms_list, input_columns=["data", "label"],
                                output_columns=["data", "label"],
                                num_parallel_workers=self.num_parallel_workers)
        
        data_set = data_set.shuffle(buffer_size=self.batch_size * 10)
        data_set = data_set.batch(self.batch_size, drop_remainder=True)
        data_set = data_set.repeat(self.repeat)
        return data_set
    
    def convert_to_gray(self):
        os.makedirs(self.dataset_ano_gray_dir)

        # convert voc color png to gray png
        print('converting voc color png to gray png ...')
        for ann in os.listdir(self.dataset_ano_dir):
            ann_im = Image.open(os.path.join(self.dataset_ano_dir, ann))
            ann_im = Image.fromarray(np.array(ann_im))
            ann_im.save(os.path.join(self.dataset_ano_gray_dir, ann))
        print('converting done')
