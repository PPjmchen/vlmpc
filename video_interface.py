import os
import json
import numpy as np
from contextlib import ExitStack

import pprint
import torch
from tools import dict_to_numpy


from hydra.utils import to_absolute_path

from vp_models.dmvfn.model.model import Model

import abc
import os



class VideoPredictionModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, batch):
        raise NotImplementedError

    def format_model_epoch_filename(self, epoch):
        raise NotImplementedError

    def get_checkpoint_file(self, checkpoint_dir, model_epoch):
        checkpoint_dir = to_absolute_path(checkpoint_dir)
        if model_epoch is not None:
            if os.path.isfile(checkpoint_dir):
                # If it's already pointing to a file, remove the file name
                checkpoint_dir = os.path.dirname(checkpoint_dir)
            checkpoint_file = os.path.join(
                checkpoint_dir, self.format_model_epoch_filename(model_epoch)
            )
        else:
            checkpoint_file = checkpoint_dir
        return checkpoint_file

    def close(self):
        """
        Clean up, e.g., any multiprocessing, or other things
        """

        pass


class DMVFNActModel(VideoPredictionModel):
    def __init__(
        self,
        checkpoint_file,
        n_past,
        action_dim=2,
        max_batch_size=800,
        epoch=None,
        device='cuda:0',
    ):
        
        
        # TODO: fix the path reading
        self.checkpoint_file = checkpoint_file
        
        self.model = Model(local_rank=0, load_path=self.checkpoint_file, training=False)

        self.num_context = n_past
        self.max_batch_size = max_batch_size
        self.base_prediction_modality = "rgb"
        self.device = device
        self.inner_bs = 10
    def prepare_batch(self, xs):
        keys = ["video", "actions"]

        xs['video'] = xs['video'] / 255. 

        batch = {
            k: torch.from_numpy(x).to(self.device, non_blocking=True).float() for k, x in xs.items() if k in keys
        }
        batch["video"] = torch.permute(batch["video"], (0, 1, 4, 2, 3))[:, :, [2, 1, 0], :, :]

        return batch
    def dict_to_float_tensor(d):
        return {
            k: torch.from_numpy(v).float() if not torch.is_tensor(v) else v
            for k, v in d.items()
    }
    def __call__(self, batch, grad_enabled=False, scale_list = [4,4,4,2,2,2,1,1,1]):
        preds = []
        batch = self.prepare_batch(batch)

        save_frames = []


        imgs =  batch['video']
        actions = batch['actions']
        img0, img1 = imgs[:, 0], imgs[:, 1]
        img0 = img0[:, [2, 1, 0], :, :]
        img1 = img1[:, [2, 1, 0], :, :]

        b, n, c, h, w = imgs.shape 
            
        action0, action1 = actions[:, 0], actions[:, 1]
        action0 = action0.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
        action1 = action1.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)

        pred_len = actions.shape[1] - 1

        save_frames.append(img0)
        save_frames.append(img1)


        loop_num = int(img0.shape[0]  / self.inner_bs)
        
        for i in range(pred_len):
            # merged = self.model.dmvfn(torch.cat((img0, img1), 1), scale = scale_list, actions = [action0, action1], training=False)
            inner_pred = []
            for inner_id in range(loop_num):
                
                with torch.no_grad():
                    merged = self.model.dmvfn(torch.cat((img0[inner_id*self.inner_bs: (inner_id+1) * self.inner_bs], img1[inner_id*self.inner_bs: (inner_id+1) * self.inner_bs]), 1), scale = scale_list, actions = [action0[inner_id*self.inner_bs: (inner_id+1) * self.inner_bs], action1[inner_id*self.inner_bs: (inner_id+1) * self.inner_bs]], training=False)
                    length = len(merged)

                if length == 0:
                    inner_pred.append(img0[inner_id*self.inner_bs: (inner_id+1) * self.inner_bs])
                else:
                    inner_pred.append(merged[-1])
            
            pred = torch.stack(inner_pred).reshape(b, c, h, w)

            preds.append(pred)
            save_frames.append(pred)
            img0 = img1
            img1 = pred

            if i != pred_len-1:
                action0 = action1
                action1 = actions[:, 2+i].unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
        
        preds.append(preds[-1])

        preds = {'rgb':torch.stack(preds, 1).permute(0, 1, 3, 4, 2)[:,:,:,:,[2, 1, 0]]}

        return dict_to_numpy(preds)


if __name__ == '__main__':
    dmvfn_model = DMVFNActModel(action_dim=2)