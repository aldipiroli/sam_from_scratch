from model.loss_functions import PixelReconstructionLoss
from model.sam import SAM
from pascal_voc_dataset.pascal_voc_dataset import PascalVOCDataset
from utils.misc import get_logger, load_config
from utils.trainer import Trainer


def train():
    config = load_config("config/sam_config.yaml")
    print(config)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    model = SAM(embed_size=256, num_output_tokens=4, num_decoder_layers=2)
    trainer.set_model(model)

    train_dataset = PascalVOCDataset(root_dir=config["DATA"]["root"], split="train")
    val_dataset = PascalVOCDataset(root_dir=config["DATA"]["root"], split="val")
    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"])
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(loss_fn=PixelReconstructionLoss())
    trainer.save_checkpoint()
    trainer.load_latest_checkpoint()
    trainer.train()


if __name__ == "__main__":
    train()
