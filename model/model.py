from vit import VisionTransformer
from loss import CosFaceLoss
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW, lr_scheduler
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader 
from torchvision import datasets
from torchvision.transforms import transforms

test_dir= r"D:\aarav\data\test"
train_dir= r"D:\aarav\data\validation"

if __name__ == "__main__":

    data_transform = transforms.Compose([
        # Resize the images to 64x64
        transforms.Resize(size=(224, 224)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])

    train_data = datasets.ImageFolder(root=train_dir, 
                                    transform=data_transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                    transform=data_transform)
    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=128, # how many samples per batch?
                                num_workers=12, # how many subprocesses to use for data loading? (higher = more)
                                shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=128,
                                num_workers=12, 
                                shuffle=False)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


    # Training setup
    num_classes = 6300
    feat_dim = 128

    vit_model = VisionTransformer(embed_dim=768, num_heads=8, mlp_dim=3072, num_layers=6, embedding_size=feat_dim)
    cosface_loss = CosFaceLoss(num_classes=num_classes, feat_dim=feat_dim)
    optimizer = AdamW(
        list(vit_model.parameters()) + list(cosface_loss.parameters()), lr=1e-4
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model.to(device)
    cosface_loss.to(device)

    num_epochs = 100

    for epoch in range(num_epochs):
        # Training phase
        vit_model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            embeddings = vit_model(images)
            loss = cosface_loss(embeddings, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation phase
        vit_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(test_dataloader, desc=f"Validation Epoch {epoch+1}"):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                embeddings = vit_model(images)
                loss = cosface_loss(embeddings, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Step the scheduler
        scheduler.step()