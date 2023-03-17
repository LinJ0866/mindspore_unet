import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore import Model, context, load_param_into_net, load_checkpoint
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
import mindspore 

from config import cfg
from src.unet.unet import UNet
from src.loss import CrossEntropyWithLogits
from src.dataset import SegDataset
from src.utils import StepLossTimeMonitor, UnetEval, TempLoss

num_class = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
             9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant',
             17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor', 21: 'edge'}

num_color = {0:'aliceblue', 1:'grey', 2:'red', 3:'green', 4:'darkorange', 5:'lime', 6:'bisque',
     7:'black', 8:'blanchedalmond', 9:'blue', 10:'blueviolet', 11:'brown', 12:'burlywood', 13:'cadetblue', 
     14:'darkorange', 15:'tan', 16:'darkviolet', 17:'cornflowerblue', 18:'yellow', 19:'crimson', 20:'darkcyan'}

color_dic = [num_color[k] for k in sorted(num_color.keys())]
bounds = list(range(21))

cmap = mpl.colors.ListedColormap(color_dic)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

class BuildEvalNetwork(nn.Cell):
    def __init__(self, network):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)

    def construct(self, input_data):
        output = self.network(input_data)
        # print(output.shape)
        output = self.softmax(output)
        # print(output.shape)
        return output

def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo

def preprocess(image):
    image_out = resize_long(image, cfg.test_crop_size)
    resize_h, resize_w, _ = image_out.shape

    image_out = (image_out - cfg.test_image_mean) / cfg.test_image_std
    pad_h, pad_w = cfg.test_crop_size - int(image_out.shape[0]), cfg.test_crop_size - int(image_out.shape[1])
    if pad_h > 0 or pad_w > 0:
        image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    
    image_out = image_out.transpose((2, 0, 1))
    return image_out, resize_h, resize_w

def infer_net():
    if cfg.model_name == 'unet':
        net = UNet()
    else:
        raise ValueError("Unsupported model: {}".format(cfg.model_name))
    
    # net = BuildEvalNetwork(net)
    # return a parameter dict for model
    param_dict = load_checkpoint(os.path.join(cfg.output_path, cfg.checkpoint_path, 'ckpt_1', cfg.checkpoint_file_path))
    # load the parameter into net
    load_param_into_net(net, param_dict)
    model = Model(net, loss_fn=TempLoss)

    image = cv2.imread('./datasets/VOC2012/JPEGImages/2007_003889.jpg')
    image, resize_h, resize_w = preprocess(image)
    res = model.predict(Tensor([image], mstype.float32))
    

    print(res[0].shape)
    
    res_out = res[0].asnumpy().argmax(0)
    print(res_out.shape)
    # print(res_out.max())
    # print(res_out)
    # print(res[0][0])
    # # res_out = res[0][0].asnumpy()
    # for i in range(21):
    #     res_out = res[0][i].asnumpy()
    print(res_out.max())
    # label_out = cv2.imdecode(np.frombuffer(res_out, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('label', res_out)
    # plt.imshow(res[0],alpha=0.8,interpolation='none', cmap=cmap, norm=norm)
    # print(res[0].shape)
    # plt.imsave('test.jpg',res_out)
    # plt.show()

    # #dataset
    # dataset_dir = 'datasets/VOC2012'
    # dataset_lst = os.path.join(dataset_dir,'ImageSets/Segmentation/val.txt')
    # dataset_img_dir = os.path.join(dataset_dir, 'JPEGImages')
    # dataset_ano_dir = os.path.join(dataset_dir,'SegmentationClass')

    # with open(dataset_lst) as f:
    #     lines = f.readlines()
    
    # for l in lines:
    #     id_ = l.strip()
    #     img_path = os.path.join(dataset_img_dir, id_ + '.jpg')
    #     label_path = os.path.join(dataset_ano_dir, id_ + '.png')
    #     with open(img_path, 'rb') as f:
    #         image = f.read()
    #         image, resize_h, resizew = preprocess(image)
    #         export(net, image, file_name=config.file_name, file_format=config.file_format)
        
    
    # model = Model(net, loss_fn=TempLoss(), metrics={"dice_coeff": dice_coeff(show_eval=config.show_eval)})

    # print("============== Starting Evaluating ============")
    # eval_score = model.eval(valid_dataset, dataset_sink_mode=False)["dice_coeff"]
    # print("============== Cross valid dice coeff is:", eval_score[0])
    # print("============== Cross valid IOU is:", eval_score[1])

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, save_graphs=False)
    infer_net()
