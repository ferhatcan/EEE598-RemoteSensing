import torch
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, PNDMScheduler, LMSDiscreteScheduler

from model import UNet

TOKEN = "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"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


if __name__ == '__main__':
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=TOKEN)
    # unet = UNet2DConditionModel(in_channels=4, out_channels=4)
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=TOKEN)
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                     num_train_timesteps=1000)

    my_unet = UNet(c_in=4, c_out=4).to(device)

    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    # text_encoder.to(device)

    vae.to(device)
    # unet.to(device)
    generator = torch.manual_seed(0)
    batch_size = 1
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    num_inference_steps = 100  # Number of denoising steps
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )

    # text_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
    #                        return_tensors="pt")
    # uncond_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    # text_embeddings = torch.cat([uncond_embeddings, uncond_embeddings])
    # text_embeddings = torch.randn((2, 77, 768)).to(device)

    latents = latents.to(device)
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    # latent_model_input = torch.cat([latents] * 2)

    # with torch.no_grad():
    # noise_pred = unet(latent_model_input, 1, encoder_hidden_states=text_embeddings).sample
        # noise_pred = torch.utils.checkpoint.checkpoint_sequential(unet, 2, (latent_model_input, 1, encoder_hidden_states=text_embeddings)).sample
    noise_pred = my_unet(latents, torch.ones(1).to(device))

    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        for _ in range(20):
            image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].show()


    tmp = 0
