# Deep Learning Final Project: Brain Tumor Detection

**Group: Angela Xu, Yoonseo Song**

## Video with Live Demo

[![Final Project Video with Live Demo](https://img.youtube.com/vi/8F-D0TqqVh0/0.jpg)](https://www.youtube.com/watch?v=8F-D0TqqVh0)

https://youtu.be/8F-D0TqqVh0

Live Demo Python Notebook:

https://colab.research.google.com/drive/1zyw99oh8VZUEDMfmwkxB4BW6GZU_zs6T?usp=sharing

## Abstract

In this project, we use CNNs to classify MRI scans into tumor and non-tumor classes. The model is trained and tested on two datasets of brain MRI images from Kaggle that we combined. We experimented with various data augmentations, network architectures, and hyperparameters to find the best model. The model achieved an accuracy of 96%, predicting 587 images out of 606 images. The findings suggest that deep learning models can accurately detect brain tumors and can be a valuable tool in clinical practice.

## Problem statement 

In the health sciences, early and accurate detection of maladies can improve the patients’ quality of life and dramatically affect the survival rate. In the recent decade, machine learning’s applications in the health sciences have allowed diseases to be detected with an accuracy never before seen. In this project, we aim to build a convolutional neural network (CNN) model that classifies whether a subject has a brain tumor based on MRI scans. 

## Related work 

We combined MRI scans from multiple Kaggle datasets to train our model to have as many unique MRI scan photos as possible.

The first dataset that we used was the brain MRI Images for brain tumor detection dataset (https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection). Data here was split into negative and positive groups based on whether there was a tumor. There were approximately 100 negative cases and 250 negative cases. 

We combined the first dataset with the Brain Tumor classification dataset (https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri. This dataset had four categories: 3 different types of tumors and one category detecting no presence of tumors. Since detection is a simpler classification problem, this posed no problem. This dataset divided its data in a training and testing set already but we chose to ignore this division. In total, this dataset had about 3500 images.

After combining these two datasets and standardizing the naming conventions of the combined dataset, we first performed preprocessing. Since the image dimensions among each dataset and across the two datasets weren’t the same, we reshape all images to 256x256 bits for training. We then created HDF5 files for each training and testing dataset so it is easier to import the processed data.

### Sample MRI Scans

![MRI Scans](https://drive.google.com/uc?export=view&id=1CAL8QedHIYVIAxYdptkiXGKaUphaTJ5M)

## Methodology 

After processing our data, we trained multiple networks to extract features from the MRI scans and classify them.

We experimented with various data augmentations, network architectures, and hyperparameters (each experiment explained in the “Experiments” section below). In each of the experiments, we graphed train loss, test loss, and test accuracy across epoch during the train. We also noted the values for final train loss, test loss, and test accuracy at the end of the training.

## Experiments/evaluation

We analyzed accuracy using a binary correct or incorrect count and used a binary cross entropy loss with a sigmoid layer for stability. 

### Experiment 1
Network architecture:
```
def __init__(self):
   super(ConvNet, self).__init__()
   self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
   self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
   self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
   self.maxpool = nn.MaxPool2d(2, 2)
   self.fc1 = nn.Linear(4096, 2)
  
 def forward(self, x):
   x = self.conv1(x)
   x = F.relu(x)
   x = self.maxpool(x)
   x = self.conv2(x)
   x = F.relu(x)
   x = self.maxpool(x)
   x = self.conv3(x)
   x = F.relu(x)
   x = self.maxpool(x)
   x = torch.flatten(x, 1)
   x = self.fc1(x)
   return x
```

Training hyperparameters:
```
BATCH_SIZE = 256
TEST_BATCH_SIZE = 10
EPOCHS = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.95
USE_CUDA = True
SEED = 0
PRINT_INTERVAL = 100
WEIGHT_DECAY = 0.0005
```

Results:
```
Train loss: 0.0463
Test loss: 0.3301
Test accuracy: 586/606 (97%)
```

![Train loss](https://drive.google.com/uc?export=view&id=1xL3_JqV4oAT4r5SmQciAWOEbXBwzIhsM)

![Test loss](https://drive.google.com/uc?export=view&id=14b93Gx8OlCBQ1pm-xkPFBuuto3JnMx38)

![Test accuracy](https://drive.google.com/uc?export=view&id=1DfSOmUHR-efAu_KYPwVByXJ2F8G0Nuph)

### Experiment 2: Dataset augmentation on train data

1. RandomHorizontalFlip

Dataset augmentation:
```
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])
```

Results:
```
Final train loss: 0.13829504549503327
Final test accuracy: 584/606 (96%)
Final test loss: 19.5082
```

![Train loss](https://drive.google.com/uc?export=view&id=1Esb5p80q88JW5KwwWM7p1PoWDSIcKVMO)

![Test loss](https://drive.google.com/uc?export=view&id=1eFP_Mg1dxG0fsrHeD59hxkiCB5U38rYB)

![Test accuracy](https://drive.google.com/uc?export=view&id=180ZWEv_3u0u_kMID7AbKDABev40kBZfB)

2. RandomAutocontrast

Dataset augmentation:
```
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomAutocontrast(0.2),
    transforms.ToTensor(),
    ])
```

Results:
```
Final train loss: 0.13918053582310677
Final test accuracy: 584/606 (96%)
Final test loss: 21.3096
```

![Train loss](https://drive.google.com/uc?export=view&id=1J5arx-sk8hl1pWJHy9W3koIv2VEgAmhK)

![Test loss](https://drive.google.com/uc?export=view&id=15ZjAqEAL7OUx5EkZEIQ_g_6-__5VyVgL)

![Test accuracy](https://drive.google.com/uc?export=view&id=1VcBSSG5ojuxD7KXegKP47QQBaSLKNYXs)

3. GaussianBlur

Dataset augmentation:
```
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomVerticalFlip(0.2),
    transforms.GaussianBlur(11),
    transforms.ToTensor(),
    ])
```

Results:
```
Final train loss: 0.1429380513727665
Final test accuracy: 584/606 (96%)
Final test loss: 0.3242
```

![Train loss](https://drive.google.com/uc?export=view&id=1hl5yPBL4YDY4zPYtWUqscNeYDvZdKZNh)

![Test loss](https://drive.google.com/uc?export=view&id=1Dt3jTIBEGcw4qBut1Eyu37sAkImBTjIF)

![Test accuracy](https://drive.google.com/uc?export=view&id=1UUbBHuvNvRb8NcpBOfsjN_NQqbNElnFS)

### Experiment 3: Network Architecture

1. 1 convolutional layer and no transformation

Network architecture:
```
def __init__(self):
   super(SimpleNet, self).__init__()
   self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
   self.maxpool = nn.MaxPool2d(2, 2)
   self.fc1 = nn.Linear(262144, 2)
  
 def forward(self, x):
   x = self.conv1(x)
   x = F.relu(x)
   x = self.maxpool(x)
   x = torch.flatten(x, 1)
   x = self.fc1(x)
   return x
```

Result: Test accuracy: 97%

![Test accuracy](https://drive.google.com/uc?export=view&id=1zH8Vvn_LWM-MicoP47hmzgNC7ckOUFjN)

2. 3 wider convolutional layer

Network architecture:
```
 def __init__(self):
   super(ComplexNet, self).__init__()
   self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
   self.conv2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
   self.conv3 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
   self.maxpool = nn.MaxPool2d(2, 2)
   self.fc1 = nn.Linear(8192, 2)
  
 def forward(self, x):
   x = self.conv1(x)
   x = F.relu(x)
   x = self.maxpool(x)
   x = self.conv2(x)
   x = F.relu(x)
   x = self.maxpool(x)
   x = self.conv3(x)
   x = F.relu(x)
   x = self.maxpool(x)
   x = torch.flatten(x, 1)
   x = self.fc1(x)
   return x
```

Result: Test accuracy: 97%

![Test accuracy](https://drive.google.com/uc?export=view&id=19r1LDfSf7BjoJrtqIljAV0933xfFOeyB)

### Experiment 4: Hyperparameter experimentation

From the default set of hyperparameters (batch size = 256, test batch size = 10, epoch = 10, learning rate = 0.01, momentum = 0.95, seed = 0, weight decay = 0.0005), we played around with the learning rate, momentum, and weight decay, we changed each of them at a time, while keeping other parameters as default. We tried learning rate of 0.01, 0.005, 0.001, momentum of 0.95, 0.90, 0.85, and weight decay of 0.0005, 0.0010, 0.0015, and graphed the train loss, test loss, and test accuracy to find the best value for each parameter.

1. Learning rate = 0.01, 0.005, 0.001

![Train loss](https://drive.google.com/uc?export=view&id=1MTqEL6JMeZAQ-PhHA8EgnxPUkaEfgtbs)

![Test loss](https://drive.google.com/uc?export=view&id=1t9BhEHIyJT20-v6cxmk6E-fLcR5CN47b)

![Test accuracy](https://drive.google.com/uc?export=view&id=1W2CslAiPoQ1aPGinR4KfUVE7W68XRDiA)

2. Momentum = 0.95, 0.90, 0.85

![Train loss](https://drive.google.com/uc?export=view&id=1cbX76kHK4HLJ8R1m-8j0lKq5WUoK6UcX)

![Test loss](https://drive.google.com/uc?export=view&id=1PaYMrw06K7wiQ2cksm-usOvxegA2wgEk)

![Test accuracy](https://drive.google.com/uc?export=view&id=14Rz8Ro-bGZWtckqflptfPa0arj5Lx2Fx)

3. Weight decay = 0.0005, 0.0010, 0.0015
   
![Train loss](https://drive.google.com/uc?export=view&id=1uRyKYPBU-uf4DTv2QX4jmcndsODXYV7u)

![Test loss](https://drive.google.com/uc?export=view&id=1x_9EjOruD2XmDuMx5lUjQW47MicayMcW)

![Test accuracy](https://drive.google.com/uc?export=view&id=1wlfjH4MQH0hbugZaRlh558zUauM8AF3R)

## Results 

Our best model was able to achieve a final accuracy of 97%. We used the hyperparameters that yielded the best results in our experiment. That is, we used a learning rate of 0.005, momentum of 0.85, weight decay of 0.0005, training batch size of 256, and a testing batch size of 10. We trained for 15 epochs, ending training before our model began overfitting.


### Final model:
Dataset augmentation: None

Network architecture:
```
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
    self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
    self.maxpool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(4096, 2)

    self.best_accuracy = -1
    
  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.maxpool(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.maxpool(x)
    x = self.conv3(x)
    x = F.relu(x)
    x = self.maxpool(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    return x
```

Training hyperparameters:
```
BATCH_SIZE = 256
TEST_BATCH_SIZE = 10
EPOCHS = 15
LEARNING_RATE = 0.005
MOMENTUM = 0.85
USE_CUDA = True
SEED = 0
PRINT_INTERVAL = 100
WEIGHT_DECAY = 0.0005
```

Results:

![Train loss](https://drive.google.com/uc?export=view&id=1f30o3bwaQzpt5KKJaN3elP4G-QxMqhOy)

![Test loss](https://drive.google.com/uc?export=view&id=1pHKQVK-Q2LHsTrpk0abT2nqxcVtuNqPn)

![Test accuracy](https://drive.google.com/uc?export=view&id=1vFlEAvjQoCV6ehgH0fNNmln8bc9Ja19z)

The findings suggest that deep learning models can accurately detect brain tumors and can be a valuable tool in clinical practice. However, an accuracy of 97%, although close to 100%, is nowhere near high enough for our model to be employed in a real world setting.

Future improvements on the model could include converting MRI images to grayscale or including more patient information inputs to the model such as age and health habits.
