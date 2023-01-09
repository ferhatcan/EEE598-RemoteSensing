import argparse
from contextlib import nullcontext
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace
from fastprogress.fastprogress import force_console_behavior
import fastprogress
from fastprogress import progress_bar
from torch import optim
from utils import *
from model import UNet_conditional, EMA, UNet_conditional_WO_Self, UNet_Classic
import logging
import wandb


# fastprogress.fastprogress.NO_BAR = True
# master_bar, progress_bar = force_console_behavior()

config = SimpleNamespace(
    run_name="UNet_classic_new_data",
    epochs=100000,
    noise_steps=100,
    seed=42,
    batch_size=3,
    img_size=64,
    num_classes=10,
    dataset_path=r"C:\GeneratedCloudDataset-Aerial\info.json",
    train_folder="train",
    val_folder="test",
    device="cuda",
    slice_size=1,
    use_wandb=False,
    do_validation=True,
    fp16=True,
    log_every_epoch=10,
    num_workers=0,
    lr=3e-4)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=3, c_out=3,
                 device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        # self.model = UNet_conditional(c_in, c_out, num_classes=num_classes, time_dim=3*64*64).to(device)
        # self.model = UNet_conditional_WO_Self(c_in, c_out, num_classes=num_classes, time_dim=3*64*64).to(device)
        self.model = UNet_Classic(c_in, c_out).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        n = labels.shape[0]
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps - 1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True, use_wandb=False):
        # all_time_stamps = list(range(1, self.noise_steps))
        avg_loss = 0.
        if train:
            self.model.train()
        else:
            self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, gts) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                gts = gts.to(self.device)
                flatten_shape = 1
                for k in range(1, len(images.shape)):
                    flatten_shape *= images.shape[k]
                images = torch.reshape(images, (images.shape[0], flatten_shape))
                # for t_ind in random.sample(all_time_stamps, int(self.noise_steps / 5)):
                # t = (torch.ones(gts.shape[0]) * t_ind).long().to(self.device)
                t = self.sample_timesteps(gts.shape[0]).to(self.device)
                x_t, noise = self.noise_images(gts, t)
                if np.random.random() < 0.0:
                    predicted_noise = self.model(x_t, t, None)
                else:
                    predicted_noise = self.model(x_t, t, images)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss.item()
                if train:
                    self.train_step(loss)
                    if use_wandb:
                        wandb.log({"train_mse": loss.item(),
                                   "learning_rate": self.scheduler.get_last_lr()[0]})
                pbar.comment = f"MSE={loss.item():2.5f}"
        return avg_loss / (i+1)

    def one_epoch_classic(self, train=True, use_wandb=False):
        avg_loss = 0.
        if train:
            self.model.train()
        else:
            self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, gts) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                gts = gts.to(self.device)

                predicted_im = self.model(images)
                loss = self.mse(gts, predicted_im)
                avg_loss += loss.item()
                if train:
                    self.train_step(loss)
                    if use_wandb:
                        wandb.log({"train_mse": loss.item(),
                                   "learning_rate": self.scheduler.get_last_lr()[0]})
                pbar.comment = f"MSE={loss.item():2.5f}"
        return avg_loss / (i+1)

    def log_images(self, use_wandb=False):
        "Log images to wandb and save them to disk"
        for i, (images, gts) in enumerate(self.val_dataloader):
            images = images.to(self.device)
            flatten_shape = 1
            for k in range(1, len(images.shape)):
                flatten_shape *= images.shape[k]
            images = torch.reshape(images, (images.shape[0], flatten_shape))
            sampled_images = self.sample(use_ema=False, labels=images)
            ema_sampled_images = self.sample(use_ema=True, labels=images)
            plot_images(sampled_images)  # to display on jupyter if available

            if use_wandb:
                wandb.log({"sampled_images": [wandb.Image(img.permute(1, 2, 0).squeeze().cpu().numpy()) for img in
                                              sampled_images]})
                wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1, 2, 0).squeeze().cpu().numpy()) for img in
                                                  ema_sampled_images]})

            break

    @staticmethod
    def convert_to_uint8(x):
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x

    def log_images_classic(self, use_wandb=False):
        "Log images to wandb and save them to disk"
        ind = random.randint(0, len(self.val_dataloader))
        with torch.no_grad():
            for i, (images, gts) in enumerate(self.val_dataloader):
                if not i == ind:
                    continue
                images = images.to(self.device)
                sampled_images = self.model(images)
                ema_sampled_images = self.ema_model(images)

                plot_images(self.convert_to_uint8(images))
                plot_images(self.convert_to_uint8(sampled_images.detach()))  # to display on jupyter if available
                plot_images(self.convert_to_uint8(ema_sampled_images.detach()))  # to display on jupyter if available
                plot_images(self.convert_to_uint8(gts.detach()))  # to display on jupyter if available

                if use_wandb:
                    wandb.log({"sampled_images": [wandb.Image(img.permute(1, 2, 0).squeeze().cpu().numpy()) for img in
                                                  sampled_images]})
                    wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1, 2, 0).squeeze().cpu().numpy()) for img in
                                                      ema_sampled_images]})

                break
        return self.convert_to_uint8(images), self.convert_to_uint8(sampled_images.detach()), \
               self.convert_to_uint8(ema_sampled_images.detach()), self.convert_to_uint8(gts.detach())

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, use_wandb=False, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        if use_wandb:
            at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional",
                                metadata={"epoch": epoch})
            at.add_dir(os.path.join("models", run_name))
            wandb.log_artifact(at)

    def prepare(self, args):
        mk_folders(args.run_name)
        device = args.device
        self.train_dataloader, self.val_dataloader = get_data(args)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.001)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr,
                                                       steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        best_loss = 1e8
        pbar = progress_bar(range(args.epochs), total=args.epochs, leave=True)
        for epoch in pbar:
            logging.info(f"Starting epoch {epoch}:\n")
            # avg_loss = self.one_epoch(train=True, use_wandb=args.use_wandb)
            avg_loss = self.one_epoch_classic(train=True, use_wandb=args.use_wandb)
            pbar.comment = f"MSE={avg_loss:2.5f}"

            ## validation
            # if args.do_validation:
            #     avg_loss = self.one_epoch(train=False, use_wandb=args.use_wandb)
            #     if args.use_wandb:
            #         wandb.log({"val_mse": avg_loss})

            # log predicitons
            # if epoch % args.log_every_epoch == 0:
            #     self.log_images(use_wandb=args.use_wandb)
            # self.save_model(run_name=args.run_name, use_wandb=args.use_wandb, epoch=epoch)

            # save model
            if epoch % args.log_every_epoch == 0:
                self.save_model(run_name=f'{args.run_name}_latest', use_wandb=args.use_wandb, epoch=epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(run_name=f'{args.run_name}_best', use_wandb=args.use_wandb, epoch=epoch)

    def inference(self, args):
        # log predicitons
        self.load(model_cpkt_path=r'models\DDPM_conditional')
        self.log_images(use_wandb=args.use_wandb)


def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--use_wandb', type=bool, default=config.use_wandb, help='use wandb')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    args = vars(parser.parse_args(args=[]))

    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


def run_all():
    parse_args(config)
    os.makedirs(f'models/{config.run_name}_latest', exist_ok=True)
    os.makedirs(f'models/{config.run_name}_best', exist_ok=True)

    ## seed everything
    set_seed(config.seed)

    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes)
    with wandb.init(project="train_sd", group="train", config=config) if config.use_wandb else nullcontext():
        diffuser.prepare(config)
        # diffuser.load(model_cpkt_path=r'models\UNet_classic_latest-130')
        diffuser.fit(config)
        # diffuser.inference(config)
        tmp = 0


def run_inference(model_path):
    parse_args(config)

    ## seed everything
    set_seed(config.seed)

    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes)
    with wandb.init(project="train_sd", group="train", config=config) if config.use_wandb else nullcontext():
        diffuser.prepare(config)
        diffuser.load(model_cpkt_path=model_path)
        # diffuser.fit(config)
        images, sampled_images, ema_sampled_images, gts = diffuser.log_images_classic()
        tmp = 0

    return images, sampled_images, ema_sampled_images, gts

# if __name__ == '__main__':
#     run_all()
#     model_cpkt_path = r"models\UNet_classic_latest-130"
#     device = 'cpu'
#
#     im_path = r'D:\Dataset\NH-HAZE\01_hazy.png'
#     im = Image.open(im_path).convert('RGB')
#     im_np = np.array(im)
#     val_transforms = torchvision.transforms.Compose([
#         T.CenterCrop(1024),
#         T.Resize(128),
#         T.ToTensor(),
#         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
#
#     im_torch = val_transforms(im).to(device).unsqueeze(0)
#     model = UNet_Classic(3, 3).to(device)
#     model.load_state_dict(torch.load(os.path.join(model_cpkt_path, "ckpt.pt")))
#
#     with torch.no_grad():
#         dehaze_im = model(im_torch)
#     dehaze_im = (dehaze_im.clamp(-1, 1) + 1) / 2
#     dehaze_im = (dehaze_im * 255).type(torch.uint8)
#     plot_images(dehaze_im)
