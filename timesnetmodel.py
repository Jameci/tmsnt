import pytorch_lightning as pl


class Timesnet(pl.LightningModule):
    def __init__(self):
        super(Timesnet, self).__init__()


    def training_step(self, batch, batch_idx):
        x, x_mask, y, y_mask = batch


    def validation_step(self):
        pass


    def configure_optimizers(self):
        pass
