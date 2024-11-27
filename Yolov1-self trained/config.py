import os
import torchvision.transforms as T


DATA_PATH = 'data'
CLASSES_PATH = os.path.join(DATA_PATH, 'classes.json')

BATCH_SIZE = 8
EPOCHS = 100
WARMUP_EPOCHS = 0
LEARNING_RATE = 1E-5

EPSILON = 1E-6
IMAGE_SIZE = (640, 640)

S = 5       # Divide each image into a SxS grid
B = 2       # Number of bounding boxes to predict
C = 1      # Number of classes in the dataset
