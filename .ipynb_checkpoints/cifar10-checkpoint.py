'''
Loads the cifar10 dataset
File #1 is meant for supervised learning,
Files #2-#5 are meant for self-/semi-supervised learning
'''


import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

SX = 32


_path = 'F:/$Daten/datasets/cifar10/cifar-10-batches-py/'


print('Loading cifar10')
_images = []
_labels = []
for fn in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
    with open(_path+fn, 'rb') as f:
        data = pkl.load(f, encoding='bytes')
        for i in range(len(data[b'data'])):
            x = data[b'data'][i].reshape((3, 32, 32)).transpose((1, 2, 0)) / 255
            _images.append(x)
            _labels.append(data[b'labels'][i])

N = len(_images)



def showimg(im):
    if im.shape[0]==3:
        im = im.transpose(1, 2, 0)
    plt.imshow(im, extent=(0,1,0,1))
    plt.axis('off')
    #plt.show()

def getimg(i):
    return _images[i]
def getlabel(i):
    return _labels[i]

def resize(img, s=24):
    im = Image.fromarray((img*255).astype('uint8'))
    im = im.resize((24, 24), resample=Image.Resampling.BILINEAR)
    img = np.array(im)/255
    return img


# Cinic-10 Dataset (similar to cifar-10 but bigger)
_path = 'F:/$Daten/datasets/cinic10/'


'''
# Folgender Teil erstellt die datei files.pkl
import os
categories= ['airplane',
   'automobile',
   'bird',
   'cat',
   'deer',
   'dog',
   'frog',
   'horse',
   'ship',
   'truck']

files = []
for kind in ['valid', 'test']:
    for c in categories:
        subpath = path + kind + '/' + c + '/'
        #print(os.listdir(subpath))
        for fn in tqdm(os.listdir(subpath)):
            files.append(subpath + fn)

with open(path + 'files.pkl', 'wb') as f:
    pkl.dump(files, f)
'''

def randomcrop(im, s=8):
    r = np.random.randint(SX-s)
    t = np.random.randint(SX-s)
    return im[r:r+s, t:t+s, :]


from PIL import Image
from torchvision.transforms import v2
v2.AutoAugmentPolicy.SVHN
augmenter = v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)


'''
with open(_path + 'files.pkl', 'rb') as f:
    _files = pkl.load(f)

N_UNSUPERVISED = len(_files)

def getunsupervised(i, augment=True):
    img = Image.open(_files[i])
    if augment: img = augmenter(img)
    return np.array(img)/255

'''



_unsupervised = None
N_UNSUPERVISED = 180000
def _loadunsupervised():
    global _unsupervised
    if _unsupervised is None:
        print('Loading cinic10')
        path = 'F:/$Daten/datasets/cinic10/'
        with open(path + 'files_train.pkl', 'rb') as f:
            _unsupervised = pkl.load(f)

_loadunsupervised()

def getunsupervised(i, augment=True):
    img = _unsupervised[i]
    if augment:
        im = Image.fromarray(img)
        im = augmenter(im)
        img = np.array(im)
    return np.array(img)/255



