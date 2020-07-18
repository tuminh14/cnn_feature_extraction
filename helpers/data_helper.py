import os
import glob
from google_drive_downloader import GoogleDriveDownloader as gdd
import cv2
class data:
    @staticmethod
    def read_data():
        '''function return 4 parameter:
            X_train,X_test,y_train,y_test'''
        X_train, y_train, X_test, y_test = [],[],[],[]
        dir = os.path.abspath(os.getcwd())
        if(os.path.isdir(dir+"/data") == False):
            gdd.download_file_from_google_drive(file_id='15ftY1DRWL6_EUYgWpHgX4rKva6IFV5w2',
                                                dest_path=dir + '/data/mnist.zip',
                                                unzip=True)
            os.remove(dir+'/data/mnist.zip')
            print('Success download and unzip data')
            print('Reading data in X_train,X_test,y_train,y_test')
            sequence_folder = glob.glob(os.path.abspath(os.getcwd()) + '/data/mnist/*/*')
            for sq in sequence_folder:
                list_images_file = glob.glob(os.path.join(sq, '*.jpg'))
                for file_name in list_images_file:
                    list_dir = file_name.split('/')
                    if (list_dir[-3] == 'testing'):
                        X_test.append(cv2.imread(file_name))
                        y_test.append(list_dir[-2])
                    else:
                        X_train.append(cv2.imread(file_name))
                        y_train.append(list_dir[-2])
            print('Read file compelete')
            return X_train, X_test, y_train, y_test
        else:
            print('Reading data in X_train,X_test,y_train,y_test')
            sequence_folder = glob.glob(os.path.abspath(os.getcwd()) + '/data/mnist/*/*')
            for sq in sequence_folder:
                list_images_file = glob.glob(os.path.join(sq, '*.jpg'))
                for file_name in list_images_file:
                    list_dir = file_name.split('/')
                    if (list_dir[-3]=='testing'):
                        X_test.append(cv2.imread(file_name))
                        y_test.append(list_dir[-2])
                    else:
                        X_train.append(cv2.imread(file_name))
                        y_train.append(list_dir[-2])
            print('Read file compelete')
            return X_train, X_test, y_train, y_test







