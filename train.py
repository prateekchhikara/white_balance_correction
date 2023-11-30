import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from model import VAE
from dataloader import ImageDataset



# Function to calculate the VAE loss
def vae_loss(x_recon, x, mu, logvar):
    beta = 0.00025 # 0.0125
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_divergence


def train_vae_with_lr_schedule(model, train_dataloader, val_dataloader, num_epochs=500, initial_lr=1e-3, save_path='best_model_low_kl_nosort.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Use StepLR scheduler to decrease learning rate every 10 epochs
    lr_scheduler_step = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            gt_img, original_img = batch
            gt_img, original_img = gt_img.to(device), original_img.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(original_img)
            loss = vae_loss(x_recon, gt_img, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Step the learning rate scheduler
        lr_scheduler_step.step()

        avg_train_loss = total_loss / len(train_dataloader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                gt_img_val, original_img_val = batch
                gt_img_val, original_img_val = gt_img_val.to(device), original_img_val.to(device)
                x_recon_val, mu_val, logvar_val = model(original_img_val)
                val_loss += vae_loss(x_recon_val, gt_img_val, mu_val, logvar_val).item()

        avg_val_loss = val_loss / len(val_dataloader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Save the model if it has the lowest validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    data_dir = "demo_images/" # directory structure: two subfolders --> GT and input
    batch_size = 64

    # Create datasets for train, val, and test
    train_dataset = ImageDataset(root_dir=data_dir, split='train')
    val_dataset = ImageDataset(root_dir=data_dir, split='val')
    

    # Create data loaders for train, val, and test
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    

    # Set the latent size according to your choice
    latent_size = 128
    vae_model = VAE(latent_size)

    # training begins
    train_vae_with_lr_schedule(vae_model, train_dataloader, val_dataloader)

