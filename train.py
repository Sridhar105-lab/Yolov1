import time
import torch
import timestamped_results_creator as tsc
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Yolov1
#from . import dataset
from dataset import VOCDataset

from utils import(
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,  
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
) 

from loss import YoloLoss


seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0 # some large number would come at expense of complexity> The actual model that paper guy implemented is by training for weeks 
EPOCHS = 100
NUM_WORKERS = 4 # controls the number of CPU workers (or processes) used for loading the data during trainin
PIN_MEMORY = True
LOAD_MODEL = True  # If you neglect it you will have floating boxes
LOAD_MODEL_FILE = "AC_detector/C100.pth.tar" # to save the model that we overfit it
IMG_DIR = "C:\\Users\\Sridhar\\OneDrive\\\Desktop\\21L105\\sem7\\DL\\Assignment\\AC_dataset\\images" 
LABEL_DIR = "C:\\Users\\Sridhar\\OneDrive\\Desktop\\21L105\\sem7\\DL\\Assignment\\\AC_dataset\\labels"

class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms
    
    def __call__(self,img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img),bboxes
            
        return img, bboxes   
        
transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave = True)
    mean_loss = []
    
    for batch_idx, (x,y) in enumerate(loop):
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out,y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update the progress bar
        loop.set_postfix(loss = loss.item())

    str3 = f"Mean loss was {sum(mean_loss)/len(mean_loss)}" 
    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    return str3
    
def main():
    model = Yolov1(split_size=7, num_boxes = 2, num_classes = 20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        # This function typically loads the weights and biases from the checkpoint into the model and optionally restores the optimizer state.
    
    train_dataset = VOCDataset(
        "AC_dataset/300examples.csv",
        transform = transform,
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR,
    )
    
    test_dataset = VOCDataset(
        "AC_dataset/test.csv", 
        transform = transform,
        img_dir=IMG_DIR,
        label_dir = LABEL_DIR,
    )
    
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size= BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
        shuffle = True,
        drop_last = False, #if the no. of examples are less than batch_size then don't drop
        #just not to ruin the gradients if there are less no. of examples
    )
    
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size= BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
        shuffle = True,
        drop_last = True, #just not to ruin the gradients if there are less no. of examples
    )
    destination = tsc.create()
    
    class_labels = ["ah64", "chinook", "cougar", "f15", "f16", "seahawk"]

    for epoch in range(EPOCHS):
        str1 = f"Epoch : {epoch + 1}"
        print(f"Epoch : {epoch + 1}")
        
        #--------------------- Comment the following till the next "hash"
        '''
         include this code only after you loaded pretrained model in LOAD_MODEL_FILE and set LOAD_MODEL = True
         if you don't have one first train a new model setting LOAD_MODEL = False which gets saved in the name of LOAD_MODEL_FILE
         Then you include it to test it
        '''
        for x, y in test_loader:
            x = x.to(DEVICE)
            for idx in range(10):  # Ensure you're in range of dataset
                bboxes = cellboxes_to_boxes(model(x))
                bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")

                # Plot image with labeled bounding boxes
                plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes,class_labels)

            import sys
            sys.exit()
        #-----------------------------

        
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold = 0.5, threshold = 0.4
        )
        
        # mean average precision for every epoch
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold = 0.5, box_format="midpoint"
        )
        
        str2 = f"Train MAP : {mean_avg_prec}" 
        print(f"{str2}")
        
        str3 = train_fn(train_loader, model, optimizer, loss_fn) 
        
        if mean_avg_prec > 0.9:
            checkpoint = {
                "state_dict" : model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            time.sleep(10)
        
        content = str1 + "\n\t" + str2 + "\n\t" + str3
        tsc.writer(content, destination)

if __name__=="__main__":
    main()