import os
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from google_drive_downloader import GoogleDriveDownloader as gdd
import config.GlobalContants as contants
import numpy as np
import pickle

class data:
    @staticmethod
    def read_data(dataName, test_split = .3, validation_split = .3,shuffle = True,random_seed = 10, batch = 16):
        '''function return 4 parameter:
            X_train,X_test,y_train,y_test'''
        dataDir = "./data"
        if(contants.dataId[dataName] is None):
            raise NameError("data name not support")
        dataId = contants.dataId[dataName]
        dir = os.path.abspath(os.getcwd())
        if(os.path.isdir(dir+"/data") == False):
            gdd.download_file_from_google_drive(file_id=dataId,
                                                dest_path=dir + '/data/data.zip',
                                                unzip=True)
            os.remove(dir+'/data/data.zip')
            print('Success download and unzip data')
            print('Reading data in X_train,X_test,y_train,y_test, X_val, y_val')
            try:
                transform = transforms.Compose([transforms.ToTensor()])
                dataset = datasets.ImageFolder(dataDir, transform=transform)
                dataset_size = len(dataset)
                indices = list(range(dataset_size))
                split = int(np.floor(test_split * dataset_size))
                if shuffle:
                    np.random.seed(random_seed)
                    np.random.shuffle(indices)
                train_indices, test_indices = indices[split:], indices[:split]
                test_sampler = SubsetRandomSampler(test_indices)
                test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=test_sampler)

                val_split = int(np.floor((validation_split / (1 - test_split)) * len(train_indices)))
                if shuffle:
                    np.random.seed(random_seed)
                    np.random.shuffle(train_indices)
                val_indices = train_indices[:val_split]
                train_indices = train_indices[val_split:]
                train_sampler = SubsetRandomSampler(train_indices)
                val_sampler = SubsetRandomSampler(val_indices)
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=train_sampler)
                val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=val_sampler)
                X_train, y_train = iter(train_loader).next()
                X_test, y_test = iter(test_loader).next()
                X_val, y_val = iter(val_loader).next()
                print('Read file compelete')
                return X_train, y_train, X_test, y_test, X_val, y_val
            except:
                print("Read data error")
        else:
            print('Reading data in X_train,X_test,y_train,y_test, X_val, y_val')
            try:
                transform = transforms.Compose([transforms.ToTensor()])
                dataset = datasets.ImageFolder(dataDir, transform=transform)
                dataset_size = len(dataset)
                indices = list(range(dataset_size))
                split = int(np.floor(test_split * dataset_size))
                if shuffle:
                    np.random.seed(random_seed)
                    np.random.shuffle(indices)
                train_indices, test_indices = indices[split:], indices[:split]
                test_sampler = SubsetRandomSampler(test_indices)
                test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=test_sampler)

                val_split = int(np.floor((validation_split / (1 - test_split)) * len(train_indices)))
                if shuffle:
                    np.random.seed(random_seed)
                    np.random.shuffle(train_indices)
                val_indices = train_indices[:val_split]
                train_indices = train_indices[val_split:]
                train_sampler = SubsetRandomSampler(train_indices)
                val_sampler = SubsetRandomSampler(val_indices)
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=train_sampler)
                val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=val_sampler)
                X_train, y_train = iter(train_loader).next()
                X_test, y_test = iter(test_loader).next()
                X_val, y_val = iter(val_loader).next()
                print('Read file compelete')
                pickle.dump((X_train, y_train, X_test, y_test, X_val, y_val), open('abc.sav', 'wb'))
                return X_train, y_train, X_test, y_test, X_val, y_val
            except Exception as e:
                print("Read data error", e)







