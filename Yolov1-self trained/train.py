import torch
import numpy as np
from models import YOLOv1ResNet
from loss import SumSquaredErrorLoss
import config
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm
from dataset import YOLODataset
import os
from datetime import datetime
from torch.utils.data import DataLoader
def save_metrics():
    np.save(os.path.join(root, 'train_losses'), train_losses)
    np.save(os.path.join(root, 'test_losses'), test_losses)
    np.save(os.path.join(root, 'train_errors'), train_errors)
    np.save(os.path.join(root, 'test_errors'), test_errors)
from torch.optim.lr_scheduler import LambdaLR
def lr_lambda(epoch):
    if epoch < 70:  
        return 0.1  
    elif epoch < 140:
        return 0.01
if __name__ =="__main__":
    with open('data.yaml', 'r') as file:
        data_config = yaml.safe_load(file)
    now = datetime.now()
    train_img_dir = data_config['train']
    train_label_dir = data_config['train']
    valid_img_dir = data_config['val']
    valid_label_dir = data_config['val']
    train_dataset = YOLODataset(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        S=5,
        B=2,
        C=1,  # Only 1 class (enemy)
        transform=None  # Replace with your transformation logic
    )
    writer = SummaryWriter()
    valid_dataset = YOLODataset(
        img_dir=valid_img_dir,
        label_dir=valid_label_dir,
        S=5,
        B=2,
        C=1,  # Only 1 class (enemy)
        transform=None
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv1ResNet().to(device)
    weight_dir = "models\yolo_v1\\11_27_2024\\03_52_28\\weights"
    # Specify the path to the pre-trained weight file
    pretrained_weight_path = os.path.join(weight_dir, 'best_weights_epoch_95.pth')  # Replace with the actual path

    #Load pre-trained weights if they exist
    if os.path.exists(pretrained_weight_path):
        model.load_state_dict(torch.load(pretrained_weight_path))
        tqdm.write(f"Loaded pre-trained weights from {pretrained_weight_path}")
    else:
        tqdm.write("No pre-trained weights found. Initializing model from scratch.")

    loss_function = SumSquaredErrorLoss()
    root = os.path.join(
            'models',
            'yolo_v1',
            now.strftime('%m_%d_%Y'),
            now.strftime('%H_%M_%S')
        )
    weight_dir = os.path.join(root, 'weights')
    if not os.path.isdir(weight_dir):
        os.makedirs(weight_dir)
    train_losses = np.empty((2, 0))
    test_losses = np.empty((2, 0))
    train_errors = np.empty((2, 0))
    test_errors = np.empty((2, 0))

    # Tạo optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-6  # LR ban đầu (phù hợp với Warm-up)
    )

    # Tạo scheduler với LambdaLR

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Vòng lặp huấn luyện
    best_val_loss = float('inf')

    for epoch in tqdm(range(config.WARMUP_EPOCHS + config.EPOCHS), desc='Epoch'):
        model.train()
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc='Train', leave=False)):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = model(data)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)

        train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)

        tqdm.write(f"Epoch {epoch + 1}/{config.WARMUP_EPOCHS + config.EPOCHS}, Train Loss: {train_loss:.4f}")

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}, Current LR: {current_lr:.2e}")

        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for data, labels in tqdm(valid_loader, desc='Test', leave=False):
                    data = data.to(device)
                    labels = labels.to(device)
                    predictions = model.forward(data)
                    loss = loss_function(predictions, labels)
                    test_loss += loss.item() / len(valid_loader)
                    del data, labels

            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
            save_metrics()

            tqdm.write(f"Epoch {epoch + 1}/{config.WARMUP_EPOCHS + config.EPOCHS}, Validation Loss: {test_loss:.4f}")

            # Save weights if validation loss improves
            if test_loss < best_val_loss:
                best_val_loss = test_loss
                torch.save(model.state_dict(), os.path.join(weight_dir, f'best_weights_epoch_{epoch + 1}.pth'))
                tqdm.write(f"Validation loss improved to {test_loss:.4f}, saving model weights.")

    save_metrics()
    torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))