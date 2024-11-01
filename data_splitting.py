import os
import numpy as np
import Config
import pickle
from sklearn.model_selection import StratifiedKFold
import shutil

def Data_Acq(fileName):
    with open(fileName, 'rb') as f:
        sample = pickle.load(f)
    return sample

def CrossValidation(dir, savedir, cross_subject_flag=True):
    print('Starting Data Splitting')
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

    # Cross-validation
    seed = 2
    np.random.seed(seed)
    num_k = Config.num_k
    kfold = StratifiedKFold(n_splits=num_k, shuffle=True, random_state=seed)

    for count, (trainIndex, testIndex) in enumerate(kfold.split(subjectList, np.zeros((len(subjectList),)))):
        print(f"Processing Fold {count}")
        fold_dir = os.path.join(savedir, f"Fold_{count}")
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        else:
            shutil.rmtree(fold_dir)
            os.makedirs(fold_dir)

        # Train
        with open(os.path.join(savedir, f"Fold_{count}", "train_list.txt"), "w") as train_file:
            for tempIndex in trainIndex:
                subjectIndex = subjectList[tempIndex]
                for tempDir in dataDir:
                    if tempDir[:3] == subjectIndex and cross_subject_flag:  # Match ID and select data
                        train_file.write(os.path.join(dir, tempDir) + "\n")

                    if tempDir == subjectIndex and cross_subject_flag is False:
                        train_file.write(os.path.join(dir, tempDir) + "\n")

        print(f"Fold {count} Train split completed")

        # Test
        with open(os.path.join(savedir, f"Fold_{count}", "test_list.txt"), "w") as test_file:
            for tempIndex in testIndex:
                subjectIndex = subjectList[tempIndex]
                for tempDir in dataDir:
                    if tempDir[:3] == subjectIndex and cross_subject_flag:  # Match ID and select data
                        test_file.write(os.path.join(dir, tempDir) + "\n")

                    if tempDir == subjectIndex and cross_subject_flag is False:
                        test_file.write(os.path.join(dir, tempDir) + "\n")

        print(f"Fold {count} Test split completed")

    print('Data Splitting completed')

if __name__ == '__main__':
    CrossValidation(Config.preprocessed_dir_savesamples, Config.savedir_train_and_test, cross_subject_flag=True)