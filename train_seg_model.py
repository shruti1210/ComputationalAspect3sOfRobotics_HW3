import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import image
import numpy as np
from random import seed
from sim import get_tableau_palette


# ==================================================
mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]
# ==================================================

class RGBDataset(Dataset):
    def __init__(self, img_dir):
        """
            Initialize instance variables.
            :param img_dir (str): path of train or test folder.
            :return None:
        """
        # TODO: complete this method
        # ===============================================================================
        self.img_dir = img_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_rgb, std_rgb),])
        self.dataset_length = len(os.listdir(self.img_dir  + 'rgb/'))
        #pass
        # ===============================================================================

    def __len__(self):
        """
            Return the length of the dataset.
            :return dataset_length (int): length of the dataset, i.e. number of samples in the dataset
        """
        # TODO: complete this method
        # ===============================================================================
        #pass
        return self.dataset_length
        # ===============================================================================

    def __getitem__(self, idx):
        """
            Given an index, return paired rgb image and ground truth mask as a sample.
            :param idx (int): index of each sample, in range(0, dataset_length)
            :return sample: a dictionary that stores paired rgb image and corresponding ground truth mask.
        """
        # TODO: complete this method
        # Hint:
        # - Use image.read_rgb() and image.read_mask() to read the images.
        # - Think about how to associate idx with the file name of images.
        # - Remember to apply transform on the sample.
        # ===============================================================================
        rgb_img = image.read_rgb(self.img_dir + 'rgb/' + f"{idx}_rgb.png")
        rgb_img = self.transform(rgb_img) 
        gt_mask = torch.LongTensor(image.read_mask(self.img_dir + 'gt/' + f"{idx}_gt.png"))
        sample = {'input': rgb_img, 'target': gt_mask}
        return sample

        # ===============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class convolutionBlock(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super(convolutionBlock, self).__init__()
        self.conv = nn.Conv2d(inputChannel, outputChannel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        x = self.conv(inputs)
        return self.relu(x)

class encoderBlock(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super(encoderBlock, self).__init__()
        self.conv = convolutionBlock(inputChannel, outputChannel)
        self.pool = nn.MaxPool2d((2,2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoderBlock(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super(decoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(inputChannel, outputChannel, kernel_size=2, stride=2, padding=0)
        self.conv = convolutionBlock(outputChannel+outputChannel, outputChannel)
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        
        return x

class miniUNet(nn.Module):
    def __init__(self):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super(miniUNet, self).__init__()
        # TODO: complete this method
        # ===============================================================================
        #pass
        self.e1 = encoderBlock(3, 16)
        self.e2 = encoderBlock(16, 32)
        self.e3 = encoderBlock(32, 64)
        self.e4 = encoderBlock(64, 128)
        
        self.b = convolutionBlock(128, 256)
        
        self.d1 = decoderBlock(256, 128)
        self.d2 = decoderBlock(128, 64)
        self.d3 = decoderBlock(64, 32)
        self.d4 = decoderBlock(32, 16)
        
        self.output = nn.Conv2d(16, 4, kernel_size=1, padding=0) 
        # ===============================================================================

    def forward(self, x):
        # TODO: complete this method
        # ===============================================================================
        #pass
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        b = self.b(p4)
        
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        output = self.output(d4)
        return output
        # ===============================================================================


def save_chkpt(model, epoch, test_miou, chkpt_path):
    """
        Save the trained model.
        :param model (torch.nn.module object): miniUNet object in this homework, trained model.
        :param epoch (int): current epoch number.
        :param test_miou (float): miou of the test set.
        :return: None
    """
    state = {'model_state_dict': model.state_dict(),
             'epoch': epoch,
             'model_miou': test_miou, }
    torch.save(state, chkpt_path)
    print("checkpoint saved at epoch", epoch)


def load_chkpt(model, chkpt_path, device):
    """
        Load model parameters from saved checkpoint.
        :param model (torch.nn.module object): miniUNet model to accept the saved parameters.
        :param chkpt_path (str): path of the checkpoint to be loaded.
        :return model (torch.nn.module object): miniUNet model with its parameters loaded from the checkpoint.
        :return epoch (int): epoch at which the checkpoint is saved.
        :return model_miou (float): miou of the test set at the checkpoint.
    """
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    model_miou = checkpoint['model_miou']
    print("epoch, model_miou:", epoch, model_miou)
    return model, epoch, model_miou


def save_prediction(model, dataloader, dump_dir, device, BATCH_SIZE):
    """
        For all datapoints d in dataloader, save  ground truth segmentation mask (as {id}.png)
          and predicted segmentation mask (as {id}_pred.png) in dump_dir.
        :param model (torch.nn.module object): trained miniUNet model
        :param dataloader (torch.utils.data.DataLoader object): dataloader to use for getting predictions
        :param dump_dir (str): dir path for dumping predictions
        :param device (torch.device object): pytorch cpu/gpu device object
        :param BATCH_SIZE (int): batch size of dataloader
        :return: None
    """
    print(f"Saving predictions in directory {dump_dir}")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    model.eval()
    with torch.no_grad():
        for batch_ID, sample_batched in enumerate(dataloader):
            data, target = sample_batched['input'].to(device), sample_batched['target'].to(device)
            output = model(data)
            _, pred = torch.max(output, dim=1)
            for i in range(pred.shape[0]):
                gt_image = convert_seg_split_into_color_image(target[i].cpu().numpy())
                pred_image = convert_seg_split_into_color_image(pred[i].cpu().numpy())
                combined_image = np.concatenate((gt_image, pred_image), axis=1)
                test_ID = batch_ID * BATCH_SIZE + i
                image.write_mask(combined_image, f"{dump_dir}/{test_ID}_gt_pred.png")


def iou(pred, target, n_classes=4):
    """
        Compute IoU on each object class and return as a list.
        :param pred (np.array object): predicted mask
        :param target (np.array object): ground truth mask
        :param n_classes (int): number of classes
        :return cls_ious (list()): a list of IoU on each object class
    """
    cls_ious = []
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(1, n_classes):  # class 0 is background
        pred_P = pred == cls
        target_P = target == cls
        pred_N = ~pred_P
        target_N = ~target_P
        if target_P.sum() == 0:
            # print("class", cls, "doesn't exist in target")  # testing (comment out later, don't delete)
            continue
        else:
            intersection = pred_P[target_P].sum()  # TP
            FP = pred_P[target_N].sum()
            FN = pred_N[target_P].sum()
            union = intersection + FN + FP  # or pred_P.sum() + target_P.sum() - intersection
            cls_ious.append(float(intersection) / float(union))
    return cls_ious


def run(model, loader, criterion, is_train=False, optimizer=None):
    """
        Run forward pass for each sample in the dataloader. Run backward pass and optimize if training.
        Calculate and return mean_epoch_loss and mean_iou
        :param model (torch.nn.module object): miniUNet model object
        :param loader (torch.utils.data.DataLoader object): dataloader 
        :param criterion (torch.nn.module object): Pytorch criterion object
        :param is_train (bool): True if training
        :param optimizer (torch.optim.Optimizer object): Pytorch optimizer object
        :return mean_epoch_loss (float): mean loss across this epoch
        :return mean_iou (float): mean iou across this epoch
    """
    model.train(is_train)
    # TODO: complete this function 
    # ===============================================================================
    mean_epoch_loss, mean_iou = 0.0, 0.0
    return mean_epoch_loss, mean_iou
    # ===============================================================================

def convert_seg_split_into_color_image(img):
    color_palette = get_tableau_palette()
    colored_mask = np.zeros((*img.shape, 3))

    print(np.unique(img))

    for i, unique_val in enumerate(np.unique(img)):
        if unique_val == 0:
            obj_color = np.array([0, 0, 0])
        else:
            obj_color = np.array(color_palette[i-1]) * 255
        obj_pixel_indices = (img == unique_val)
        colored_mask[:, :, 0][obj_pixel_indices] = obj_color[0]
        colored_mask[:, :, 1][obj_pixel_indices] = obj_color[1]
        colored_mask[:, :, 2][obj_pixel_indices] = obj_color[2]
    return colored_mask.astype(np.uint8)

def train(model, device, train_loader, criterion, optimizer):
    """
    Loop over each sample in the dataloader.
    Do forward + backward + optimize procedure. Compute average sample loss and mIoU on the dataset.
    """
    model.train()
    train_loss, train_iou = 0, 0
    # TODO
    total_loss = 0 
    total_miou = 0 
    num_samples = 0
    for i, batch in enumerate(train_loader):
        inputs = batch['input']
        targets = batch['target']        
        inputs=inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        miou = iou(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item() * inputs.size(0)
        total_miou += np.mean(miou) * inputs.size(0)
        num_samples += inputs.size(0)
    train_loss = total_loss/num_samples
    train_iou = total_miou/num_samples
    return train_loss, train_iou

import torch.utils.data as data

if __name__ == "__main__":
    # ==============Part 4 (a) Training Segmentation model ================
    # Complete all the TODO's in this file
    # - HINT: Most TODO's in this file are exactly the same as homework 2.

    seed(0)
    torch.manual_seed(0)
    # Check if GPU is being detected
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # TODO: Prepare train and test datasets
    # Load the "dataset" directory using RGBDataset class as a pytorch dataset
    # Split the above dataset into train and test dataset in 9:1 ratio using `torch.utils.data.random_split` method
    
    root_dir = './dataset/'
    dataset = RGBDataset(root_dir)
    # Split dataset into train and test sets
    train_dataset, test_dataset = data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])



    # TODO: Prepare train and test Dataloaders. Use appropriate batch size
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    # TODO: Prepare model
    model = miniUNet().to(device)

    # TODO: Define criterion, optimizer and learning rate scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # TODO: Train and test the model. 
    # Tips:
    # - Remember to save your model with best mIoU on objects using save_chkpt function
    # - Try to achieve Test mIoU >= 0.9 (Note: the value of 0.9 only makes sense if you have sufficiently large test set)
    # - Visualize the performance of a trained model using save_prediction method. Make sure that the predicted segmentation mask is almost correct.

    train_loss_list, train_miou_list, val_loss_list, val_miou_list = list(), list(), list(), list()
    epoch, max_epochs = 1, 30  # TODO: you may want to make changes here
    best_miou = float('-inf')
    while epoch <= max_epochs:
        print('Epoch (', epoch, '/', max_epochs, ')')
        print("train_loader", train_loader)
        train_loss, train_miou = train(model, device, train_loader, criterion, optimizer)
        train_loss_list.append(train_loss)
        train_miou_list.append(train_miou)
        print('Train loss & mIoU: %0.2f %0.2f' % (train_loss, train_miou))
    
    


    
