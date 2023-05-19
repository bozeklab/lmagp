from lightning.pytorch import LightningModule
from collections import OrderedDict
import torch

# in the current implementation metrics are calculated per gpu.
# validation_step_end and test_step_end should be implemented to aggregate the outputs of each 
# gpu that will be passed to validation_epoch_end and test_epoch_end.
# see https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
class LitModel(LightningModule):
    def __init__(self, model=None, optimizer_config=None, loss_function=None, log_sync_dist=False, log_batch_size=None, keep_output_struct=False, val_step_metrics = [], val_epoch_metrics = [], test_step_metrics = [], test_epoch_metrics = []):
        super(LitModel, self).__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.loss_function = loss_function
        self.log_sync_dist = log_sync_dist
        self.log_batch_size = log_batch_size
        self.val_step_metrics = val_step_metrics
        self.val_epoch_metrics = val_epoch_metrics
        self.test_step_metrics = test_step_metrics
        self.test_epoch_metrics = test_epoch_metrics
        self.keep_output_struct = keep_output_struct # some models output more than just logits, and a cross-entropy like loss function doesnt need that
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def configure_optimizers(self):
        self.optimizer_config = dict(self.optimizer_config)
        self.optimizer_config['optimizer'] = \
            self.optimizer_config['optimizer'](self.model.parameters())
        if self.optimizer_config.get('lr_scheduler') is not None:
            self.optimizer_config['lr_scheduler'] = dict(self.optimizer_config['lr_scheduler'])
            self.optimizer_config['lr_scheduler']['scheduler'] = \
                self.optimizer_config['lr_scheduler']['scheduler'](self.optimizer_config['optimizer'])
        return self.optimizer_config

    def training_step(self, batch, batch_nb):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        if isinstance(y_hat, (tuple, list, OrderedDict)) and not self.keep_output_struct:
            y_hat = y_hat[0]
        loss = self.loss_function(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True, batch_size=self.log_batch_size)
        return loss

    def _val_test_step_impl(self, batch, batch_nb, step_metrics, dataloader_idx=0):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        if isinstance(y_hat, (tuple, list, OrderedDict)):
            y_hat = y_hat[0]
        metrics_dict = {}
        for metric_dict in step_metrics:
            metric_key, metric_func = list(metric_dict.items())[0]
            metric_value = metric_func(y_hat, y)
            metrics_dict[metric_key] = metric_value
        self.log_dict(metrics_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.log_sync_dist, logger=True, batch_size=self.log_batch_size)
        return {'y_hat': y_hat, 'y': y}

    def _val_test_epoch_end_impl(self, steps_outputs, epoch_metrics, dataloader_idx=0):
        all_y_hat = torch.cat([x['y_hat'] for x in steps_outputs])
        all_y = torch.cat([x['y'] for x in steps_outputs])
        metrics_dict = {}
        for metric_dict in epoch_metrics:
            metric_key, metric_func = list(metric_dict.items())[0]
            metric_key += '/{}'.format(dataloader_idx)
            metric_value = metric_func(all_y_hat, all_y)
            metrics_dict[metric_key] = metric_value
        self.log_dict(metrics_dict, sync_dist=self.log_sync_dist, logger=True, add_dataloader_idx=True, batch_size=self.log_batch_size) #add_dataloader_idx=False: for some reason when set to True (default) the idx does not get added, so I do it myself some lines above

    def validation_step(self, batch, batch_nb, dataloader_idx=0):
        metrics = self._val_test_step_impl(batch, batch_nb, self.val_step_metrics, dataloader_idx)
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs
        if isinstance(validation_step_outputs[0], list):
            for idx, val_dataloader_step_outputs in enumerate(validation_step_outputs):
                self._val_test_epoch_end_impl(val_dataloader_step_outputs, self.val_epoch_metrics, idx)
        else:
            self._val_test_epoch_end_impl(validation_step_outputs, self.val_epoch_metrics)
        self.validation_step_outputs = []

    def test_step(self, batch, batch_nb, dataloader_idx=0):
        metrics = self._val_test_step_impl(batch, batch_nb, self.test_step_metrics, dataloader_idx)
        self.test_step_outputs.append(metrics)
        return metrics
    
    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        if isinstance(test_step_outputs[0], list):
            for idx, test_dataloader_step_outputs in enumerate(test_step_outputs):
                self._val_test_epoch_end_impl(test_dataloader_step_outputs, self.test_epoch_metrics, idx)
        else:
            self._val_test_epoch_end_impl(test_step_outputs, self.test_epoch_metrics)
        self.test_step_outputs = []
