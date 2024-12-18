import os
import Config
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader
import numpy as np
from logger import TrainingLogger
from Model.deep_learning import DatasetLoad, CNN_Model, Train_one_epoch, Evaluate, \
    Get_mean_and_std, Recod_and_Save_Train_Detial, FNN_Model
from evaluation_metrics import Metrix_computing
from utils.losses import LabelSmoothingLoss, mixup_data, mixup_criterion
import math


import warnings
warnings.filterwarnings("ignore")


def Load_data(dir):
    data_dir = []
    with open(dir, 'r') as file:
        line = file.readline()
        while line:
            data_dir.append(line.strip())
            line = file.readline()

    return data_dir


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def Main(feature_name, model_name):
    logger = TrainingLogger("logs")
    logger.log("=== Starting Training Pipeline ===")
    logger.log(f"Feature Type: {feature_name} (Spectrogram/Mel-Spectrogram/Statistical Features)")
    logger.log(f"Model Architecture: {model_name} (Convolutional Neural Network/Feed-Forward Neural Network)")
    
    # Device setup
    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device} (CUDA Graphics Processing Unit/Central Processing Unit)")
    
    # Create directories
    if not os.path.exists(os.path.join(Config.savedir_train_and_test, "Record")):
        os.makedirs(os.path.join(Config.savedir_train_and_test, "Record"))
        logger.log("Created Record directory for saving model checkpoints")

    # Initialize metric lists
    logger.log("Initializing metric tracking lists:")
    logger.log("- SEN: Sensitivity (True Positive Rate)")
    logger.log("- SPE: Specificity (True Negative Rate)")
    logger.log("- ACC: Accuracy (Correct Predictions/Total Predictions)")
    logger.log("- AE: Average Error")
    logger.log("- HS: Harmonic Score")
    senList, speList, aeList, hsList, accList, confusion_matrix_List = [], [], [], [], [], []

    # Process each fold
    existing_folds = [f for f in os.listdir(Config.savedir_train_and_test) if f.startswith("Fold_")]
    existing_folds.sort(key=lambda x: int(x.split("_")[1]))
    logger.log(f"Found {len(existing_folds)} cross-validation folds")

    for fold in existing_folds:
        logger.log(f"\n=== Processing {fold} ===")
        count = int(fold.split("_")[1])
        train_list_path = os.path.join(Config.savedir_train_and_test, fold, "train_list.txt")
        test_list_path = os.path.join(Config.savedir_train_and_test, fold, "test_list.txt")
        
        if not os.path.exists(train_list_path) or not os.path.exists(test_list_path):
            print(f"Train or test list not found for {fold}. Skipping...")
            continue

        print(f"Processing {fold}")

        # Data Load
        train_data_dir_list = Load_data(train_list_path)
        test_data_dir_list = Load_data(test_list_path)

        # Normalization using mean & std
        mean, std = Get_mean_and_std(DatasetLoad(train_data_dir_list, feature_name))
        input_transform = None
        if "statistics" in feature_name:
            input_transform = [mean, std]
        else:
            input_transform = Compose([ToTensor(), Normalize(mean, std)])
        print(f"Fold: {count}, mean: {mean}, std: {std}")

        # Train & Test
        train_dataloader = DataLoader(
            dataset=DatasetLoad(train_data_dir_list, feature_name, input_transform),
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_data = DataLoader(
            dataset=DatasetLoad(test_data_dir_list, feature_name, input_transform),
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Model load
        model = None
        if model_name == "CNN":
            model = CNN_Model(Config.Num_classes).to(device)
        elif model_name == "FNN":
            model = FNN_Model(Config.Num_classes).to(device)

        # Load existing weights if available
        weight_path = os.path.join(Config.savedir_train_and_test, "Record", f"Fold_{count}_model_weights_Epoch_Final.pth")
        start_epoch = 0
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path))
            print(f"Loaded weights from {weight_path}")
            # Set start_epoch to the last saved epoch
            start_epoch = Config.EPOCH - (Config.EPOCH % 30)  # Adjust this based on your saving logic

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=Config.INITIAL_LR, 
                                    weight_decay=Config.WEIGHT_DECAY)

        num_training_steps = len(train_dataloader) * Config.NUM_EPOCHS
        num_warmup_steps = len(train_dataloader) * Config.WARMUP_EPOCHS
        
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                  num_warmup_steps, 
                                                  num_training_steps)

        # train & test
        train_recorder, test_recorder = [], []
        for epoch in range(start_epoch, Config.EPOCH):
            train_info = Train_one_epoch(model, optimizer, train_dataloader, device, epoch, logger, scheduler)
            train_recorder.append(train_info)

            test_info = Evaluate(model, test_data, device)
            test_recorder.append(test_info)

            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            logger.log(f"Current Learning Rate: {current_lr}")

            # Log detailed metrics
            logger.log(f"\nEpoch {epoch} Summary:")
            logger.log(f"Training - Loss: {train_info[0]:.4f}, Acc: {train_info[1]:.4f}")
            logger.log(f"Validation - Loss: {test_info[0]:.4f}, Acc: {test_info[1]:.4f}")
            logger.log(f"Confusion Matrix:\n{test_info[6]}")
            logger.log(f"Classification Report:\n{test_info[7]}\n")

            print(f"Epoch: {epoch+1}/{Config.EPOCH}: Train Loss: {train_info[0]:.6f}, ACC: {train_info[1]:.6f}, "
                  f"SEN: {train_info[2]:.6f}, SPE: {train_info[3]:.6f}, AE: {train_info[4]:.6f}, HS: {train_info[5]:.6f}")
            print(f"Epoch: {epoch + 1}/{Config.EPOCH}: Test  Loss: {test_info[0]:.6f}, ACC: {test_info[1]:.6f}, "
                  f"SEN: {test_info[2]:.6f}, SPE: {test_info[3]:.6f}, AE: {test_info[4]:.6f}, HS: {test_info[5]:.6f}\n")

            # save the weights
            if (epoch + 1) % 30 == 0 and epoch > 0:
                print(test_info[-2])
                print(test_info[-1])
                torch.save(model.state_dict(), os.path.join(Config.savedir_train_and_test, "Record", f"Fold_{count}_model_weights_Epoch_{epoch}.pth"))

        # Save the final model
        torch.save(model.state_dict(), os.path.join(Config.savedir_train_and_test, "Record", f"Fold_{count}_model_weights_Epoch_Final.pth"))
        # Plot picture & save the training record
        Recod_and_Save_Train_Detial(count, Config.savedir_train_and_test,
                                    train_recorder, test_recorder, show_fucntion=False)

        # Test
        test_info = Evaluate(model, test_data, device)
        print(f"Fold: {count}\n"
              f"Final Test  Loss: {test_info[0]:.6f}, ACC: {test_info[1]:.6f}, "
              f"SEN: {test_info[2]:.6f}, SPE: {test_info[3]:.6f}, AE: {test_info[4]:.6f}, HS: {test_info[5]:.6f}")
        print(test_info[-2], "\n", test_info[-1], "\n\n\n")

        accList.append(test_info[1])
        senList.append(test_info[2])
        speList.append(test_info[3])
        aeList.append(test_info[4])
        hsList.append(test_info[5])
        confusion_matrix_List.append(test_info[6])

    # Final Results
    logger.log("\n=== Final Cross-Validation Results ===")
    logger.log(f"Mean Accuracy: {np.mean(accList):.4f} (Average across all folds)")
    logger.log(f"Mean Sensitivity: {np.mean(senList):.4f} (Average True Positive Rate)")
    logger.log(f"Mean Specificity: {np.mean(speList):.4f} (Average True Negative Rate)")
    logger.log(f"Mean Average Error: {np.mean(aeList):.4f}")
    logger.log(f"Mean Harmonic Score: {np.mean(hsList):.4f}")

    for i, matrix in enumerate(confusion_matrix_List):
        print(i)
        print(matrix, "\n")

    print("acc:", accList)
    print("sen：", senList)
    print("spe:", speList)
    print("ae :", aeList)
    print("he :", hsList)
    print("mean acc:", np.mean(accList))
    print("mean sen:", np.mean(senList))
    print("mean spe:", np.mean(speList))
    print("mean ae :", np.mean(aeList))
    print("mean he :", np.mean(hsList))


def Evaluate(model, data_loader, device):
    all_labels, all_prediction = [], []
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.float().to(device), labels.to(device)
            outputs = model(inputs)
            # Extract the 'output' from the dictionary returned by the model
            outputs = outputs['output']
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_prediction.extend(predicted.cpu().numpy())

    confusion_matrix, sen, spe, ae, hs, acc, report = Metrix_computing(all_labels, all_prediction)

    # Return a tuple instead of numpy array
    return 0.0, acc, sen, spe, ae, hs, confusion_matrix, report


if __name__ == '__main__':
    feature_name_list = ["spectrogram", "mel_spectrogram", "statistics_feature"]
    feature_name = feature_name_list[1]

    model_list = ["CNN", "FNN"]
    mode_name = model_list[0]

    Main(feature_name, mode_name)



