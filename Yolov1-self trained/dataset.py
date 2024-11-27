import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import config 
from tqdm import tqdm

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, S=5, B=2, C=1, transform=None, back = False):
        """
        Args:
            img_dir (str): Path to the directory containing images.
            label_dir (str): Path to the directory containing labels (YOLO format).
            S (int): The grid size (e.g., 7x7).
            B (int): Number of bounding boxes per grid cell.
            C (int): Number of classes.
            transform (callable, optional): Transform function for preprocessing.
        """
        self.back=back
        self.img_dir = img_dir + "\\images"
        self.label_dir = label_dir + "\\labels"
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.transform2 = transforms.Compose([
                transforms.ToTensor()
            ])
        self.S = S
        self.B = B
        self.C = C
        self.images = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get the image path and corresponding label path
        img_name = self.images[index]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Load the bounding boxes from the label file
        boxes = []
        with open(label_path) as f:
            for line in f.readlines():
                class_label, x_center, y_center, width, height = map(float, line.strip().split())
                boxes.append([int(class_label), x_center, y_center, width, height])
        boxes = torch.tensor(boxes)
        if(self.back==False):
            image = self.transform(image)
        else:
            image = self.transform2(image)

        # Create the label matrix
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        bbox_counters = torch.zeros((self.S, self.S), dtype=torch.int32)
        for box in boxes:
            class_label, x_center, y_center, width, height = box.tolist()
            class_label = int(class_label)

            # Determine the cell (i, j) in the grid
            i, j = int(self.S * y_center), int(self.S * x_center)
            x_cell, y_cell = self.S * x_center - j, self.S * y_center - i
            width_cell, height_cell = width , height 

            # If no object already exists in cell (i, j)
            if bbox_counters[i][j] < self.B:
                # Mark that an object exists in this cell
                bbox_index = bbox_counters[i][j]
                label_matrix[i, j, self.C] = 1

                # Add bounding box coordinates (relative to cell)
                label_matrix[i, j, bbox_index*5+self.C :bbox_index*5+self.C + 5] = torch.tensor(
                    [1, x_cell, y_cell, width_cell, height_cell]
                )

                # Set one-hot encoding for the class
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
