import mlflow.experiments
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import mlflow

# Define the generator network
class Generator(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, output_size=784):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=1):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# Hyperparameters
batch_size = 64
lr = 0.0002
num_epochs = 20
latent_size = 100

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Initialize the generator and discriminator
generator = Generator(input_size=latent_size)
discriminator = Discriminator()

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

mlflow.set_experiment("GAN model - MNIST-final")

mlflow.set_tracking_uri("https://dagshub.com/atikul-islam-sajib/Advanced-Software-Engineering.mlflow")

with mlflow.start_run() as run:
    # Training loop
    for epoch in range(num_epochs):
        netG_loss = []
        netD_loss = []
        for i, (images, _) in enumerate(dataloader):
            batch_size = images.size(0)
            images = images.view(batch_size, -1)
            
            # Create labels
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train discriminator
            outputs = discriminator(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            z = torch.randn(batch_size, latent_size)
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            d_loss = d_loss_real + d_loss_fake
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Train generator
            z = torch.randn(batch_size, latent_size)
            fake_images = generator(z)
            outputs = discriminator(fake_images)

            g_loss = criterion(outputs, real_labels)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            
            netG_loss.append(d_loss.item())
            netD_loss.append(g_loss.item())

            if (i+1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}],\
                    Step [{i+1}/{len(dataloader)}],\
                        D Loss: {d_loss.item():.4f},\
                            G Loss: {g_loss.item():.4f},\
                                D(x): {real_score.mean().item():.2f},\
                                    D(G(z)): {fake_score.mean().item():.2f}')
                
        
        mlflow.log_metrics({
            "netG_loss{}".format(epoch+1): np.mean(netD_loss),
            "netD_loss{}".format(epoch+1) : np.mean(netG_loss)
        })
        
        mlflow.log_metric("netG_loss", np.mean(netG_loss), step=epoch+1)
        mlflow.log_metric("netD_loss", np.mean(netD_loss), step=epoch+1)
        
        mlflow.pytorch.log_model(generator, "neG{}".format(epoch+1))
        mlflow.pytorch.log_model(discriminator, "netD{}".format(epoch+1))
        
        fake_samples = torch.randn(batch_size, latent_size)
        predict = generator(fake_samples)
        predict = predict[0:4]
        save_image(predict, os.path.join("C:/Users/atiku/OneDrive/Desktop/DVC/Advanced-Software-Engineering/image/{}.png".format(epoch+1)))
        
    
    mlflow.log_artifact("C:/Users/atiku/OneDrive/Desktop/DVC/Advanced-Software-Engineering/image")
        
        
    
    mlflow.log_params({
        "epochs": num_epochs,
        "lr": lr,
        "beta1": 0.5,
        "beta2": 0.999,
        "batch_size": batch_size,
        "latent_space": latent_size,
        "optimizerD": "Adam",
        "optimizerG": "Adam",
        "criterion": "BCE"
    })

# # Save generated images for each epoch
# import os

# # Ensure the 'generated_images' directory exists
# os.makedirs("generated_images", exist_ok=True)

# with torch.no_grad():
#     z = torch.randn(batch_size, latent_size)
#     fake_images = generator(z)
#     fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
#     grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
    
#     # Plot and save the image
#     plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
#     plt.axis('off')  # Hide the axes for better visualization
#     image_path = f"generated_images/epoch_{epoch+1}.png"
#     plt.savefig(image_path)
#     plt.close()
    
#     # Log the image using MLflow
#     mlflow.log_artifact(image_path)
