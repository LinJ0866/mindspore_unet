import os

import mindspore.nn as nn
from mindspore import Model, context
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
import mindspore 

from config import cfg
from src.unet.unet import UNet
from src.loss import CrossEntropyWithLogits
from src.dataset import SegDataset
from src.utils import StepLossTimeMonitor

def train_net():
    if cfg.model_name == 'unet':
        net = UNet()
    else:
        raise ValueError("Unsupported model: {}".format(cfg.model_name))
    
    #dataset
    train_dataset = SegDataset('VOC2012', 'train')
    train_dataset = train_dataset.get_mindrecord_dataset()

    train_data_size = train_dataset.get_dataset_size()
    print("dataset length is:", train_data_size)
    
    # loss
    criterion = CrossEntropyWithLogits()

    # optimizer
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    # loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(cfg.FixedLossScaleManager, False)
    amp_level = "O0" if cfg.device_target == "GPU" else "O3"
    model = Model(net, loss_fn=criterion, optimizer=optimizer,
                  amp_level=amp_level)

    ckpt_save_dir = os.path.join(cfg.output_path, cfg.checkpoint_path, 'ckpt_1')
    save_ck_steps = train_data_size * cfg.epochs
    ckpt_config = CheckpointConfig(save_checkpoint_steps=save_ck_steps,
                                   keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_{}_adam'.format(cfg.model_name),
                                 directory=ckpt_save_dir,
                                 config=ckpt_config)

    print("============== Starting Training ==============")
    callbacks = [StepLossTimeMonitor(batch_size=cfg.train_batch_size, per_print_times=cfg.per_print_times), ckpoint_cb]
    # if config.run_eval:
    #     eval_model = Model(UnetEval(net, need_slice=need_slice, eval_activate=config.eval_activate.lower()),
    #                        loss_fn=TempLoss(), metrics={"dice_coeff": dice_coeff(False, config.show_eval)})
    #     eval_param_dict = {"model": eval_model, "dataset": valid_dataset, "metrics_name": config.eval_metrics}
    #     eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
    #                            eval_start_epoch=config.eval_start_epoch, save_best_ckpt=True,
    #                            ckpt_directory=ckpt_save_dir, besk_ckpt_name="best.ckpt",
    #                            metrics_name=config.eval_metrics)
    #     callbacks.append(eval_cb)
    model.train(int(cfg.epochs / cfg.repeat), train_dataset, callbacks=callbacks, dataset_sink_mode=cfg.dataset_sink_mode)

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, save_graphs=False)
    train_net()
