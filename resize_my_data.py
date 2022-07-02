import config
import os
import cv2


if __name__ == '__main__':
    path = config.DATA_PATH + '/omniglot'
    for folder in os.listdir(path):
        sub_p = path + '/' + folder
        for sub_folder in os.listdir(sub_p):
            ssp = sub_p + '/' + sub_folder
            for file in os.listdir(ssp):
                img = cv2.imread(ssp+'/'+file)
                re = cv2.resize(img, dsize=(28, 28))
                cv2.imwrite(ssp+'/'+file, re)
