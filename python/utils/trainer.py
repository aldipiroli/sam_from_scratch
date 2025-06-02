from pathlib import Path

import torch
import torch.nn.functional as F
from model.loss_functions import compute_iou_between_masks
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.misc import downsample_mask, get_device, get_prompt_from_gtmask, plot_mask_predictions


class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.epoch = 0
        self.pred_threshold = 0.5
        self.img_size = config["MODEL"]["img_size"]

        self.ckpt_dir = Path(config["CKPT_DIR"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device()
        self.artifacts_img_dir = Path(config["IMG_OUT_DIR"])
        self.artifacts_img_dir.mkdir(parents=True, exist_ok=True)

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
        if self.optim_config["optimizer"] == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.optim_config["lr"])

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

    def prepare_inputs(self, gt_masks, deterministic=False):
        selected_prompts, prompt_gt_mask, selected_classes = get_prompt_from_gtmask(
            gt_masks, deterministic=deterministic
        )
        selected_prompts = selected_prompts.float().to(self.device)
        prompt_gt_mask = prompt_gt_mask.to(self.device)
        prompt_gt_mask = prompt_gt_mask.unsqueeze(1)  # (b, 1, h, w)

        selected_prompts_norm = torch.zeros_like(selected_prompts)
        selected_prompts_norm[..., 0] = selected_prompts[..., 0] / self.img_size[1]
        selected_prompts_norm[..., 1] = selected_prompts[..., 1] / self.img_size[2]
        return selected_prompts_norm, prompt_gt_mask

    def postprocessor(self, mask_pred, iou_pred, apply_threshold=False):
        mask_pred = torch.sigmoid(mask_pred)
        iou_pred = torch.sigmoid(iou_pred)

        if apply_threshold:
            mask_pred = (mask_pred > self.pred_threshold).float()
        return mask_pred, iou_pred

    def upsample_preds(self, mask_pred):
        mask_pred = F.interpolate(
            mask_pred.unsqueeze(1).float(), size=(self.img_size[1], self.img_size[2]), mode="nearest"
        ).squeeze(1)
        return mask_pred

    def train(self):
        for curr_epoch in range(self.optim_config["num_epochs"]):
            self.epoch = curr_epoch
            self.train_one_epoch()
            self.evaluate_model()

    def train_one_epoch(self, eval_every_iter=150):
        self.model.train()
        with tqdm(enumerate(self.train_loader), desc=f"Epoch {self.epoch}") as pbar:
            for n_iter, (data, all_gt_masks) in pbar:
                self.optimizer.zero_grad()
                data = data.to(self.device)
                all_gt_masks = all_gt_masks.to(self.device)

                prompts_norm, prompt_gt_masks = self.prepare_inputs(all_gt_masks)
                pred_masks, pred_ious = self.model(data, prompts_norm)
                loss = self.loss_fn(gt_masks=prompt_gt_masks, pred_masks=pred_masks, pred_iou=pred_ious)
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({"loss": loss.item()})
                if (n_iter + 1) % eval_every_iter == 0:
                    self.evaluate_model()
        self.save_checkpoint()

    def evaluate_model(self, max_num_iter=100, max_plot_iter=3):
        self.logger.info("Running Evaluation...")
        self.model.eval()
        all_iou = []
        for n_iter, (data, all_gt_masks) in enumerate(self.val_loader):
            if n_iter > max_num_iter:
                break

            self.optimizer.zero_grad()
            data = data.to(self.device)
            all_gt_masks = all_gt_masks.to(self.device)

            prompts_norm, prompt_gt_masks = self.prepare_inputs(all_gt_masks)
            pred_masks, pred_ious = self.model(data, prompts_norm)

            # compute iou eval
            gt_masks_down = downsample_mask(all_gt_masks, target_dim=(pred_masks.shape[-1], pred_masks.shape[-1]))
            actual_iou = compute_iou_between_masks(gt_masks_down, pred_masks)
            all_iou.append(actual_iou)
            if n_iter < max_plot_iter:
                self.plot_predictions(
                    all_gt_masks=all_gt_masks,
                    prompt_gt_masks=prompt_gt_masks,
                    data=data,
                    pred_masks=pred_masks,
                    pred_ious=pred_ious,
                    prompts_norm=prompts_norm,
                    batch_id=0,
                    i=n_iter,
                )

        self.logger.info(f"Epoch {self.epoch} avg IoU {torch.mean(torch.tensor(all_iou)).item():.2f}")
        self.model.train()

    def overfit_one_batch(self):
        self.model.train()
        it = iter(self.train_loader)
        data, all_gt_masks = next(it)
        # data, all_gt_masks = get_simple_data(1, 224, 224)
        for i in range(1000000):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            all_gt_masks = all_gt_masks.to(self.device)

            prompts_norm, prompt_gt_masks = self.prepare_inputs(all_gt_masks, deterministic=False)
            pred_masks, pred_ious = self.model(data, prompts_norm)
            loss = self.loss_fn(gt_masks=prompt_gt_masks, pred_masks=pred_masks, pred_ious=pred_ious)
            # print(f"Loss {loss.item():.6f}")
            loss.backward()
            self.optimizer.step()
            if (i % 10) == 0:
                # pred_masks, pred_ious = self.postprocessor(pred_masks, pred_ious, apply_threshold=True)
                self.plot_predictions(
                    all_gt_masks=all_gt_masks,
                    prompt_gt_masks=prompt_gt_masks,
                    data=data,
                    pred_masks=pred_masks,
                    pred_ious=pred_ious,
                    prompts_norm=prompts_norm,
                    batch_id=0,
                    i=0,
                )

        self.save_checkpoint()

    def gradient_sanity_check(self):
        total_gradient = 0
        no_grad_name = []
        grad_name = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                no_grad_name.append(name)
                self.logger.info(f"None grad: {name}")
            else:
                grad_name.append(name)
                total_gradient += torch.sum(torch.abs(param.grad))
        assert total_gradient == total_gradient
        if len(no_grad_name) > 0:
            self.logger.info(f"no_grad_name {no_grad_name}")
            raise ValueError("layers without gradient are present")
        assert len(no_grad_name) == 0

    def plot_predictions(
        self, all_gt_masks, prompt_gt_masks, data, pred_masks, pred_ious, prompts_norm, batch_id=0, i=0
    ):
        prompt_gt_masks_down = downsample_mask(prompt_gt_masks, target_dim=(pred_masks.shape[-1], pred_masks.shape[-2]))
        actual_iou = compute_iou_between_masks(prompt_gt_masks_down, pred_masks)
        plot_mask_predictions(
            image=data[batch_id],
            all_gt_masks=all_gt_masks[batch_id],
            prompt_gt_masks=prompt_gt_masks[batch_id],
            prompt_gt_masks_down=prompt_gt_masks_down[batch_id],
            pred_masks=pred_masks[batch_id],
            pred_ious=pred_ious[batch_id],
            actual_iou=actual_iou[batch_id],
            prompt=prompts_norm[batch_id],
            filename=f"{self.artifacts_img_dir}/tmp_{str(i).zfill(4)}.png",
        )
