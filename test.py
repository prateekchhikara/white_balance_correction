import torch
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader
from wand.image import Image as WandImage
import numpy as np
import cv2
from model import VAE
from dataloader import ImageDataset

# Model Inference
def apply_sharpening(batch_tensor, radius=1.0, sigma=0.5):
    save_folder = "images_temp"
    for i in range(batch_tensor.shape[0]):
        
        image = batch_tensor[i].permute(1, 2, 0)  # Change from (3, 256, 256) to (256, 256, 3)
        image = (image * 255).clamp(0, 255).to(torch.uint8)  # Convert to uint8
        image = np.array(image)
        image_pil = Image.fromarray(image)
        image_pil.save(os.path.join(save_folder, f"image_{i + 1}.png"))

    sharpened_images = []
    for i in range(batch_tensor.shape[0]):
        filename = os.path.join(save_folder, f"image_{i + 1}.png")
        with WandImage(filename=filename) as img:
            # img.sharpen(radius=radius, sigma=sigma)
            enhanced = np.array(img)
            enhanced = cv2.bilateralFilter(enhanced, 75, 16, 16)
            enhanced = enhanced.transpose((2, 0, 1)) / 255.0  # Change back to (3, 256, 256) and normalize
            sharpened_images.append(enhanced)

    sharpened_images = np.stack(sharpened_images)

    return sharpened_images




def inference_and_plot(model, test_dataloader, num_images=5):
    print("Plotting and Saving results")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load the saved model checkpoint
    checkpoint_path = 'best_model_low_kl.pth'  # Update with your saved model path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= num_images:
                break

            gt_img, original_img = batch
            gt_img, original_img = gt_img.to(device), original_img.to(device)

            # Perform inference
            x_recon, _, _ = model(original_img)

            # Move data back to CPU for plotting
            gt_img = gt_img.cpu()
            original_img = original_img.cpu()
            x_recon = x_recon.cpu()

            x_recon_sm = apply_sharpening(x_recon, 8, 4)
            x_recon_sm = torch.from_numpy(x_recon_sm)


            # Plot the images
            plt.figure(figsize=(40, 55))
            plt.subplot(1, 4, 1)
            plt.imshow(vutils.make_grid(gt_img, nrow=4, normalize=True).permute(1, 2, 0))
            plt.title("Ground Truth", fontdict={'fontsize': 20})

            plt.subplot(1, 4, 2)
            plt.imshow(vutils.make_grid(original_img, nrow=4, normalize=True).permute(1, 2, 0))
            plt.title("Original Image", fontdict={'fontsize': 20})

            plt.subplot(1, 4, 3)
            plt.imshow(vutils.make_grid(x_recon, nrow=4, normalize=True).permute(1, 2, 0))
            plt.title("Reconstructed Image", fontdict={'fontsize': 20})

            plt.subplot(1, 4, 4)
            plt.imshow(vutils.make_grid(x_recon_sm, nrow=4, normalize=True).permute(1, 2, 0))
            plt.title("Reconstructed Smooth Image", fontdict={'fontsize': 20})

            plt.savefig("enhanced_output.png")
            plt.show()
            return original_img, x_recon, x_recon_sm
        

if __name__ == "__main__":
    data_dir = "demo_images/" # directory structure: two subfolders --> GT and input
    batch_size = 64

    # Set the latent size according to your choice
    latent_size = 128
    vae_model = VAE(latent_size)
    test_dataset = ImageDataset(root_dir=data_dir, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    original_img, x_recon, x_recon_sm = inference_and_plot(vae_model, test_dataloader, num_images=5)
