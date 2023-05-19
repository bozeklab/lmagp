import lightning.pytorch as pl
import omegaconf

class LitDataloadersContainer(pl.LightningDataModule):
    def __init__(self, 
        train_dataloader = None,
        val_dataloader = None,
        test_dataloader = None,
        predict_dataloader = None,
    ):
        super().__init__()
        self.train_dataloader = self.get_dataloader_impl(train_dataloader)
        self.val_dataloader = self.get_dataloader_impl(val_dataloader)
        self.test_dataloader = self.get_dataloader_impl(test_dataloader)
        self.predict_dataloader = self.get_dataloader_impl(predict_dataloader)

    def prepare_data(self):
        return

    def setup(self, stage = None):
        return
    
    def get_dataloader_impl(self, dataloader):
        def dataloader_impl():
            return omegaconf.OmegaConf.to_container(dataloader) if isinstance(dataloader, omegaconf.basecontainer.BaseContainer) else dataloader
        if dataloader is None:
            return None
        else:
            return dataloader_impl
