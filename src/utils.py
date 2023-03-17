import time
import numpy as np

import mindspore.nn as nn
from mindspore.ops import operations as ops
from mindspore.train.callback import Callback
from mindspore.common.tensor import Tensor

class StepLossTimeMonitor(Callback):
    def __init__(self, batch_size, per_print_times=1):
        super(StepLossTimeMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.batch_size = batch_size

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        step_seconds = time.time() - self.step_time
        step_fps = self.batch_size * 1.0 / step_seconds

        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        self.losses.append(loss)
        if self._per_print_times != 0 and cur_step_in_epoch % self._per_print_times == 0:
            # TEST
            print("step: %s, loss is %s, fps is %s" % (cur_step_in_epoch, loss, step_fps), flush=True)

    def epoch_begin(self, run_context):
        self.epoch_start = time.time()
        self.losses = []

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_cost = time.time() - self.epoch_start
        step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        step_fps = self.batch_size * 1.0 * step_in_epoch / epoch_cost
        print("epoch: {:3d}, avg loss:{:.4f}, total cost: {:.3f} s, per step fps:{:5.3f}".format(
            cb_params.cur_epoch_num, np.mean(self.losses), epoch_cost, step_fps), flush=True)

class UnetEval(nn.Cell):
    """
    Add Unet evaluation activation.
    """

    def __init__(self, net, need_slice=False, eval_activate="softmax"):
        super(UnetEval, self).__init__()
        self.net = net
        self.need_slice = need_slice
        self.transpose = ops.Transpose()
        self.softmax = ops.Softmax(axis=-1)
        self.argmax = ops.Argmax(axis=-1)
        self.squeeze = ops.Squeeze(axis=0)
        if eval_activate.lower() not in ("softmax", "argmax"):
            raise ValueError("eval_activate only support 'softmax' or 'argmax'")
        self.is_softmax = True
        if eval_activate == "argmax":
            self.is_softmax = False

    def construct(self, x):
        out = self.net(x)
        if self.need_slice:
            out = self.squeeze(out[-1:])
        out = self.transpose(out, (0, 2, 3, 1))
        if self.is_softmax:
            softmax_out = self.softmax(out)
            return softmax_out
        argmax_out = self.argmax(out)
        return argmax_out

class TempLoss(nn.Cell):
    """A temp loss cell."""

    def __init__(self):
        super(TempLoss, self).__init__()
        self.identity = ops.identity()

    def construct(self, logits, label):
        return self.identity(logits)