from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import ceil

def load_images_from_folder(folder, gray_scale=False, size=(32,32), prefix=None):
    imgs = []
    for file in listdir(folder): 
        if isfile(join(folder,file)) and (prefix is None or file.lower().startswith(prefix.lower())):
            img = cv2.imread(join(folder, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray_scale else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(cv2.resize(img, size))
        
    return np.array(imgs)

def plot_images(images, labels=[], figsize=(15,5), rows=2):
    n = len(images)
    if len(labels) == 0:
        labels = np.arange(n)
    
    columns = ceil(n / rows)
    
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.5)
    for i,(img, label) in enumerate(zip(images, labels)):
        plt.subplot(rows, columns, i+1)
        plt.title(label)
        plt.imshow(img)
    plt.show()

def images_to_2d(images):
    return images.reshape(images.shape[0], -1)

def vector_to_image(x, resolution):
    return x.reshape(resolution[0], resolution[1], -1)/255

identity = lambda x: x

def add_bias_parameter(x):
    '''Adiciona os termos independentes (parâmetro bias) aos dados'''
    return np.concatenate([np.ones(x.shape), x], axis=1)