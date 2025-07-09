import torch
import torch.nn.functional as F
from diffusion.models.unet import UNet
from diffusion.data.loaders import get_mnist_loaders
from diffusion.diffusion.forward import ForwardDiffusion
from tqdm import tqdm
import os


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    diffusion = ForwardDiffusion(timesteps=1000)
    dataloader = get_mnist_loaders(batch_size=128)

    for epoch in range(1):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for x, _ in progress_bar:
            x = x.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device)
            noise = torch.randn_like(x)
            noisy_x = diffusion.q_sample(x, t, noise)

            pred_noise = model(noisy_x)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/unet_mnist.pth")
    print("Model saved to checkpoints/unet_mnist.pth")


if __name__ == "__main__":
    train()
