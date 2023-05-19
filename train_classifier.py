import lightning.pytorch as pl
import hydra
from hydra_zen import instantiate 

@hydra.main(config_path=None)
def task(cfg):
    pl.seed_everything(42)
    obj = instantiate(cfg)
    if obj.trainer.logger is not None:
        obj.trainer.logger.log_hyperparams(dict(cfg))
    obj.trainer.fit(obj.module, datamodule=obj.datamodule)
    if obj.get('test') and obj.datamodule.test_dataloader:
        obj.trainer.test(obj.module, datamodule=obj.datamodule, ckpt_path='best')
    return 0

if __name__ == '__main__':
    task()