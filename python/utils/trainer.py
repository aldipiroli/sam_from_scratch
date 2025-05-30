from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.misc import get_device


class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.epoch = 0

        self.ckpt_dir = Path(config["CKPT_DIR"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device()

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.logger.info("Model:")
        self.logger.info(self.model)

    def save_checkpoint(self):
        model_path = Path(self.ckpt_dir) / f"ckpt_{str(self.epoch).zfill(4)}.pt"
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            model_path,
        )
        self.logger.info(f"Saved checkpoint in: {model_path}")

    def load_latest_checkpoint(self):
        if not self.ckpt_dir.exists():
            self.logger.info("No checkpoint directory found.")
            return None

        ckpt_files = sorted(self.ckpt_dir.glob("ckpt_*.pt"))
        if not ckpt_files:
            self.logger.info("No checkpoints found.")
            return None

        latest_ckpt = max(ckpt_files, key=lambda x: int(x.stem.split("_")[1]))
        self.logger.info(f"Loading checkpoint: {latest_ckpt}")

        checkpoint = torch.load(latest_ckpt, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        return latest_ckpt

    def set_dataset(self, train_dataset, val_dataset, data_config):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.data_config = data_config

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=data_config["batch_size"],
            shuffle=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
        )
        self.logger.info(f"Train Dataset: {self.train_dataset}")
        self.logger.info(f"Val Dataset: {self.val_dataset}")

    def set_optimizer(self, optim_config):
        self.optim_config = optim_config
        if self.optim_config["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optim_config["lr"])

            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.optim_config.get("step_size", self.optim_config["weight_decay_step"]),
                gamma=self.optim_config.get("gamma", self.optim_config["weight_decay"]),
            )
        else:
            raise ValueError("Unknown optimizer")

        self.logger.info(f"Optimizer: {self.optimizer}")

    def set_loss_function(self, loss_fn):
        self.loss_fn = loss_fn.to(self.device)
        self.logger.info(f"Loss function {self.loss_fn}")

    def train(self):
        for curr_epoch in range(self.optim_config["num_epochs"]):
            self.epoch = curr_epoch
            self.train_one_epoch()
            self.evaluate_model()

    def train_one_epoch(self, eval_every_iter=500):
        self.model.train()
        with tqdm(enumerate(self.train_loader), desc=f"Epoch {self.epoch}") as pbar:
            for n_iter, (data, labels) in pbar:
                self.optimizer.zero_grad()
                data = data.to(self.device)
                output_dict = self.model(data)
                loss = self.loss_fn(output_dict)
                loss.backward()
                self.gradient_sanity_check()
                self.optimizer.step()
                pbar.set_postfix({"loss": loss.item()})
                if n_iter % eval_every_iter == 0:
                    self.evaluate_model(n_iter=n_iter)
        self.save_checkpoint()

    def evaluate_model(self):
        self.model.eval()
        for idx, (data, labels) in enumerate(self.val_loader):
            data = data.to(self.device)
            output_dict = self.model(data)
            # do stuff...

        self.model.train()

    def gradient_sanity_check(self):
        total_gradient = 0
        no_grad_name = []
        grad_name = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                no_grad_name.append(name)
                print(f"None grad: {name}")
            else:
                grad_name.append(name)
                total_gradient += torch.sum(torch.abs(param.grad))
        assert total_gradient == total_gradient
        assert len(no_grad_name) == 0
