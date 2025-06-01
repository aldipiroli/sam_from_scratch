from model.loss_functions import SAMLoss
from model.sam import SAM
from pascal_voc_dataset.pascal_voc_dataset import PascalVOCDataset
from utils.misc import get_logger, load_config
from utils.trainer import Trainer


def train():
    config = load_config("config/sam_config.yaml")
    print(config)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    model_cfg = config["MODEL"]
    model = SAM(
        embed_size=model_cfg["embed_size"],
        num_output_tokens=model_cfg["num_output_tokens"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        num_frequencies=model_cfg["num_frequencies"],
        dropout=model_cfg["dropout"],
    )
    trainer.set_model(model)

    train_dataset = PascalVOCDataset(root_dir=config["DATA"]["root"], split="train")
    val_dataset = PascalVOCDataset(root_dir=config["DATA"]["root"], split="val")
    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"])
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(loss_fn=SAMLoss())
    trainer.save_checkpoint()
    trainer.load_latest_checkpoint()
    trainer.train()
    # trainer.overfit_one_batch()


if __name__ == "__main__":
    train()
