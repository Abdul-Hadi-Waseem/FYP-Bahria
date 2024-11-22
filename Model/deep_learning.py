import os
import torch
from torch import nn
from torch.utils.data import Dataset
import sklearn
import time
import numpy as np
import sklearn.metrics as metrics
import pickle
import matplotlib.pyplot as plt
import Config
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
from torch.optim.optimizer import Optimizer
from evaluation_metrics import Metrix_computing
from utils.augmentations import AudioAugmentation
import random
from logger import TrainingLogger
import psutil

logger = TrainingLogger("logs")


class FNN_Model(nn.Module):
    def __init__(self, Num_classes):
        super(FNN_Model, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(102, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.output = nn.Sequential(
            nn.Linear(16, Num_classes)
        )

    def forward(self, x):
        logger.log(f"Forward pass - Input shape: {x.shape}", level='debug')
    
        x = self.conv1(x)
        logger.log(f"After conv1: {x.shape}", level='debug')
    
        x = self.conv2(x)
        logger.log(f"After conv2: {x.shape}", level='debug')
    
        x = self.conv3(x)
        logger.log(f"After conv3: {x.shape}", level='debug')
    
        x = self.adaptive_pool(x)
        logger.log(f"After adaptive_pool: {x.shape}", level='debug')
    
        x = torch.flatten(x, 1)
        logger.log(f"After flatten: {x.shape}", level='debug')
    
        embedding = self.classifier[:-1](x)
        output = self.classifier[-1](embedding)
    
        return {"embedding": embedding, "output": output}



class CNN_Model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Model, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Add shape logging
        #print(f"Input shape: {x.shape}")
        
        x = self.conv1(x)
        #print(f"After conv1: {x.shape}")
        
        x = self.conv2(x)
        #print(f"After conv2: {x.shape}")
        
        x = self.conv3(x)
        #print(f"After conv3: {x.shape}")
        
        x = self.adaptive_pool(x)
        #print(f"After adaptive_pool: {x.shape}")
        
        x = torch.flatten(x, 1)
        #print(f"After flatten: {x.shape}")
        
        embedding = self.classifier[:-1](x)
        output = self.classifier[-1](embedding)
        
        return {"embedding": embedding, "output": output}



def Train_one_epoch(model, optimizer, train_dataloader, device, epoch, logger, scheduler=None):
    """
    Train model for one epoch
    """
    logger.log("Setting model to train mode")
    model.train()
    optimizer.zero_grad()

    task_ce_criterion = nn.CrossEntropyLoss().to(device)
    acclosses = AverageMeter()
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    all_prediction, all_labels = [], []

    logger.log(f"Starting epoch {epoch}")

    for batch, (data, labels) in enumerate(train_dataloader):
        # Log batch progress periodically
        if batch % 10 == 0:
            logger.log(f"Epoch {epoch} - Processing batch {batch}/{len(train_dataloader)}")
            logger.log(f"Input shape: {data.shape}")

        try:
            # Move data to device
            logger.log(f"Moving batch {batch} to device", level='debug')
            data, labels = data.float().to(device), labels.long().to(device)

            # Forward pass
            logger.log(f"Forward pass for batch {batch}", level='debug')
            outputs = model(data)["output"]
            
            # Calculate loss
            logger.log(f"Computing loss for batch {batch}", level='debug')
            loss = task_ce_criterion(outputs, labels)
            acclosses.update(loss.item(), data.shape[0])

            # Backward pass
            logger.log(f"Backward pass for batch {batch}", level='debug')
            optimizer.zero_grad()
            loss.backward()
            
            # Optimizer step
            logger.log(f"Optimizer step for batch {batch}", level='debug')
            optimizer.step()

            # Scheduler step if it exists
            if scheduler is not None:
                scheduler.step()
                logger.log(f"Current learning rate: {optimizer.param_groups[0]['lr']}", level='debug')

            # Calculate accuracy
            logger.log(f"Computing accuracy for batch {batch}", level='debug')
            train_acc_sum += (outputs.argmax(dim=1) == labels).sum().cpu().item()
            n += len(labels)

            # Store predictions
            _, prediction = torch.max(outputs.data, dim=1)
            all_prediction.extend(prediction.to('cpu'))
            all_labels.extend(labels.to('cpu'))

            if batch % 10 == 0:
                logger.log(f"Completed batch {batch}/{len(train_dataloader)} - "
                         f"Loss: {loss.item():.4f}, "
                         f"Acc: {train_acc_sum/n:.4f}")

        except Exception as e:
            logger.log(f"Error in batch {batch}: {str(e)}", level='error')
            logger.log(f"Data shape: {data.shape}", level='error')
            logger.log(f"Labels shape: {labels.shape}", level='error')
            raise e

    # Compute final metrics
    logger.log("Computing final metrics for epoch")
    train_loss_sum = acclosses.avg
    confusion_matrix, sen, spe, ae, hs, acc, report = Metrix_computing(all_labels, all_prediction)

    # Log epoch results
    logger.log(f"Epoch {epoch} completed - Loss: {acclosses.avg:.4f}, Acc: {acc:.4f}")

    return acclosses.avg, acc, sen, spe, ae, hs, confusion_matrix, report
@torch.no_grad()
def Evaluate(model, data_loader, device):
    logger.log("\n=== Starting Evaluation ===")
    logger.log("Setting model to evaluation mode (disabling dropout/batch normalization)")
    
    all_labels, all_prediction = [], []
    with torch.no_grad():
        logger.log("Disabled gradient computation for evaluation")
        
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 10 == 0:
                logger.log(f"Processing batch {batch_idx}/{len(data_loader)}")
                
            inputs, labels = batch
            inputs, labels = inputs.float().to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs['output']
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_prediction.extend(predicted.cpu().numpy())

    logger.log("Computing evaluation metrics:")
    confusion_matrix, sen, spe, ae, hs, acc, report = Metrix_computing(all_labels, all_prediction)
    
    logger.log(f"Confusion Matrix:\n{confusion_matrix}")
    logger.log(f"Classification Report:\n{report}")
    
    return 0.0, acc, sen, spe, ae, hs, confusion_matrix, report




def Normalized_confusion_matrix(confusion_matrix):
    return normalize(confusion_matrix, axis=1, norm='l1')




class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def Recod_and_Save_Train_Detial(count, dir, train_recorder, test_recorder, show_fucntion=False):
    # Loss, ACC, SEN, SPE
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    train_sen, test_sen = [], []
    train_spe, test_spe = [], []
    for i in range(len(train_recorder)):
        train_loss.append(train_recorder[i][0])
        train_acc.append(train_recorder[i][1])
        train_sen.append(train_recorder[i][2])
        train_spe.append(train_recorder[i][3])

        test_loss.append(test_recorder[i][0])
        test_acc.append(test_recorder[i][1])
        test_sen.append(test_recorder[i][2])
        test_spe.append(test_recorder[i][3])


    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(6, 8))

    for i, record in enumerate([("loss", train_loss, test_loss), ("acc", train_acc, test_acc),
                                ("sen", train_sen, test_sen), ("spe", train_spe, test_spe)]):

        ax[i].set_title(record[0])
        ax[i].plot(record[1], label="Train")
        ax[i].plot(record[2], color="red", label="Test")
        ax[i].legend()


    # 画图
    plt.subplots_adjust(hspace=0.5, top=0.95, bottom=0.05)
    plt.savefig(f"{dir}\\Record\\Fold_{count}_train_detail.jpg")
    if show_fucntion == True:
        plt.show()
    else:
        plt.clf()

    # 记录存储
    record = {
        "train_recorder": train_recorder,
        "test_recorder": test_recorder
    }

    with open(f"{dir}\\Record\\Fold_{count}_train_detail_record.dat", 'wb') as f:
        pickle.dump(record, f)

    return



class DatasetLoad(Dataset):
    def __init__(self, data_dir, feature_name, input_transform=None, is_training=False):
        self.data = data_dir
        self.feature_name = feature_name
        self.input_transform = input_transform
        self.is_training = is_training
        self.augmenter = AudioAugmentation() if is_training else None

    def __getitem__(self, index):
        samples = self.Data_Acq(self.data[index])
        data = samples[self.feature_name]
        label = samples["label"]

        # Ensure data is in the correct format for CNN
        if len(data.shape) == 2:  # If 2D spectrogram
            data = data[None, :, :]  # Add channel dimension
        elif len(data.shape) == 3 and data.shape[0] > 1:  # If multi-channel
            data = data[0:1, :, :]  # Take first channel only

        # Apply augmentation during training
        if self.is_training and random.random() < 0.5:
            if random.random() < 0.5:
                data = self.augmenter.time_shift(data)
            if random.random() < 0.5:
                data = self.augmenter.add_noise(data)
            if random.random() < 0.5:
                data = self.augmenter.change_pitch(data)

        # Convert to tensor if not already
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        return data, label

    def Data_Acq(self, dir):
        file = open(dir, 'rb')
        sample = pickle.load(file, encoding='latin1')
        file.close()
        return sample

    def __len__(self):
        return len(self.data)



# Compute the mean and std value of train dataset.
def Get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # # For 语谱图
    if "spectrogram" in dataset.feature_name:
        mean = torch.zeros(1)
        std = torch.zeros(1)
        print("Spectrogram based")
        print('==> Computing mean and std..')
        for inputs, targets in dataloader:
            mean += inputs.mean()
            std += inputs.std()

        mean.div_(len(dataset))
        std.div_(len(dataset))

        return mean, std
    else:
        print("Statistics_feature")
        print('==> Computing mean and std..')

        value = torch.zeros((1, 102))
        for inputs, targets in dataloader:
            value = torch.concat((value, inputs))

        value = value[1:]
        mean = value.mean(dim=0)
        std = value.std(dim=0)

        return mean.cpu().numpy() , std.cpu().numpy()



def Load_data(dir):
    data_dir = []
    with open(dir, 'r') as file:
        line = file.readline()
        while line:
            data_dir.append(line.strip())
            line = file.readline()

    return data_dir



def print_memory_stats(logger):
    if torch.cuda.is_available():
        logger.log(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        logger.log(f"GPU Memory cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    process = psutil.Process()
    logger.log(f"CPU Memory used: {process.memory_info().rss/1024**2:.2f} MB")



if __name__ == '__main__':

    count = 0
    train_data_dir_list = Load_data(f"{Config.savedir_train_and_test}\\Fold_{count}\\train_list.txt")
    print(len(train_data_dir_list))

    nor, crk, wheeze, both = 0, 0, 0, 0
    dataloader = torch.utils.data.DataLoader(
        DatasetLoad(train_data_dir_list, "signal"),
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    for data, label in dataloader:
        if label == 0:
            nor += 1
        elif label == 1:
            crk += 1
        elif label == 2:
            wheeze += 1
        elif label == 3:
            both += 1

    print(nor, crk, wheeze, both)



    test_data_dir_list = Load_data(f"{Config.savedir_train_and_test}\\Fold_{count}\\test_list.txt")
    print(len(test_data_dir_list))
    nor, crk, wheeze, both = 0, 0, 0, 0
    dataloader = torch.utils.data.DataLoader(
        DatasetLoad(test_data_dir_list, "signal"),
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    for data, label in dataloader:
        if label == 0:
            nor += 1
        elif label == 1:
            crk += 1
        elif label == 2:
            wheeze += 1
        elif label == 3:
            both += 1

    print(nor, crk, wheeze, both)






