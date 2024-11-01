import os
import numpy as np
import Config
import pickle
from sklearn.model_selection import StratifiedKFold
import shutil
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics



def Data_Acq(fileName):
    with open(fileName, 'rb') as f:
        sample = pickle.load(f)
    return sample


def CrossValidation(dir, savedir, cross_subject_flag=True):
    dataDir = os.listdir(dir)

    # Cross_subject
    subjectList = []

    if cross_subject_flag:
        for tempDir in dataDir:
            tempDir = tempDir[:3]
            if tempDir not in subjectList:
                subjectList.append(tempDir)
    else:
        for tempDir in dataDir:
            if tempDir not in subjectList:
                subjectList.append(tempDir)

    # Cross-validation setup
    seed = 2
    np.random.seed(seed)
    num_k = Config.num_k
    kfold = StratifiedKFold(n_splits=num_k, shuffle=True, random_state=seed)

    for count, (trainIndex, testIndex) in enumerate(kfold.split(subjectList, np.zeros((len(subjectList),)))):
        print(count, "Start")
        fold_dir = os.path.join(savedir, f"Fold_{count}")
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        else:
            shutil.rmtree(fold_dir)
            os.makedirs(fold_dir)

        # Train data file creation
        with open(os.path.join(savedir, f"Fold_{count}", "train_list.txt"), "w") as train_file:
            for tempIndex in trainIndex:
                subjectIndex = subjectList[tempIndex]
                for tempDir in dataDir:
                    if tempDir[:3] == subjectIndex and cross_subject_flag:  # Match subject ID and select data
                        train_file.write(os.path.join(dir, tempDir) + "\n")
                    
                    if tempDir == subjectIndex and cross_subject_flag is False:
                        train_file.write(os.path.join(dir, tempDir) + "\n")

        print(count, "Train  over")

        # Test
        with open(os.path.join(savedir, f"Fold_{count}", "test_list.txt"), "w") as test_file:
            for tempIndex in testIndex:
                subjectIndex = subjectList[tempIndex]
                for tempDir in dataDir:
                    if tempDir[:3] == subjectIndex and cross_subject_flag:  # 匹配id，选择数据
                        test_file.write(os.path.join(dir, tempDir) + "\n")

                    if tempDir == subjectIndex and cross_subject_flag is False:
                        test_file.write(os.path.join(dir, tempDir) + "\n")

        print(count, "Test  over")
        print(count, "Over")




def Metrix_computing(y_true, y_pred):
    confusion_matrix_result = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    num_classes = confusion_matrix_result.shape[0]
    
    if num_classes > 2:
        # Multi-class metrics
        sen = np.mean([confusion_matrix_result[i, i] / np.sum(confusion_matrix_result[i, :]) for i in range(num_classes)])
        spe = np.mean([(np.sum(confusion_matrix_result) - np.sum(confusion_matrix_result[i, :]) - np.sum(confusion_matrix_result[:, i]) + confusion_matrix_result[i, i]) / 
                       (np.sum(confusion_matrix_result) - np.sum(confusion_matrix_result[i, :])) for i in range(num_classes)])
        ae = np.sum(np.diag(confusion_matrix_result)) / np.sum(confusion_matrix_result)
        hs = 2 * sen * spe / (sen + spe) if (sen + spe) > 0 else 0
        acc = ae  # For multi-class, accuracy is the same as average efficiency
    else:
        # Binary classification metrics
        if confusion_matrix_result.size == 4:
            tn, fp, fn, tp = confusion_matrix_result.ravel()
            sen = tp / (tp + fn) if (tp + fn) > 0 else 0
            spe = tn / (tn + fp) if (tn + fp) > 0 else 0
            ae = (tp + tn) / (tp + tn + fp + fn)
            hs = 2 * sen * spe / (sen + spe) if (sen + spe) > 0 else 0
            acc = ae
        else:
            # Handle the case where we don't have a 2x2 confusion matrix
            print("Warning: Unexpected confusion matrix shape for binary classification.")
            sen = spe = ae = hs = acc = 0
    
    return confusion_matrix_result, sen, spe, ae, hs, acc, report





if __name__ == '__main__':
    CrossValidation(Config.preprocessed_dir_savesamples, Config.savedir_train_and_test, cross_subject_flag=True)
