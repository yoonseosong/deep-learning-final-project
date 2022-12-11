# Deep Learning Final Project: Brain Tumor Detection

Group: Angela Xu, Yoonseo Song

For your final project you should explore any topic you are interested in related to deep learning. This could involve training a model for a new task, building a new dataset, improving deep models in some way and testing on standard benchmarks, etc. You project should probably involve some implementation, some data, and some training. The amount of effort and time should be approximately 2 homework assignments.

- website describing your project
- 2-3 minute video. 
  - problem setup
  - data used 
  - techniques, etc
  - description of which components were from preexisting work (i.e. code from github) and which components were implemented for the project (i.e. new code, gathered dataset, etc).

## Video with Live Demo

<insert>

## Abstract

In the health sciences, early and accurate detection of maladies can improve the patients’ quality of life and dramatically affect the survival rate. In the recent decade, machine learning’s applications in the health sciences have provided a more accurate and efficient way of detecting diseases. In this project, we aim to build a convolutional neural network (CNN) to classify brain MRI images into tumor and non-tumor classes. 

The model is trained and tested on two datasets of brain MRI images from Kaggle that we combined. We experimented with various data augmentations, network architectures, and hyperparameters to find the best model. The model achieved an accuracy of 96%, predicting <insert> images out of 606 images. The findings suggest that deep learning models can accurately detect brain tumors and can be a valuable tool in clinical practice.

## Problem statement 

In the health sciences, early and accurate detection of maladies can improve the patients’ quality of life and dramatically affect the survival rate. In the recent decade, machine learning’s applications in the health sciences have allowed diseases to be detected with an accuracy never before seen. In this project, we aim to build a convolutional neural network (CNN) model that classifies whether a subject has a brain tumor based on MRI scans. 

## Related work 

We combined MRI scans from multiple Kaggle datasets to train our model to have as many unique MRI scan photos as possible.

The first dataset that we used was the brain MRI Images for brain tumor detection dataset (https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection). Data here was split into negative and positive groups based on whether there was a tumor. There were approximately 100 negative cases and 250 negative cases. 

We combined the first dataset with the Brain Tumor classification dataset (https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri. This dataset had four categories: 3 different types of tumors and one category detecting no presence of tumors. Since detection is a simpler classification problem, this posed no problem. This dataset divided its data in a training and testing set already but we chose to ignore this division. In total, this dataset had about 3500 images.

After combining these two datasets and standardizing the naming conventions of the combined dataset, we first performed preprocessing. Since the image dimensions among each dataset and across the two datasets weren’t the same, we reshape all images to 256x256 bits for training. We then created HDF5 files for each training and testing dataset so it is easier to import the processed data.

## Methodology 

We trained convolutional neural networks, using convolutional layers, maxpooling layers, and fully connected layers, to extract features from the MRI scans and classify them.

We experimented with various data augmentations, network architectures, and hyperparameters (each experiment explained in the “Experiments” section below). In each of the experiments, we graphed train loss, test loss, and test accuracy across epoch during the train. We also noted the values for final train loss, test loss, and test accuracy at the end of the training.

## Experiments/evaluation

We experimented with various data augmentations, network architectures, and hyperparameters to find the best model.


### Experiment 1
Network architecture:
```
def __init__(self):
   super(AudreyNet, self).__init__()
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
<insert graphs>

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

### Experiment 4: Hyperparameter experimentation

From the default set of hyperparameters (batch size = 256, test batch size = 10, epoch = 10, learning rate = 0.01, momentum = 0.95, seed = 0, weight decay = 0.0005), we played around with the learning rate, momentum, and weight decay, we changed each of them at a time, while keeping other parameters as default. We tried learning rate of 0.01, 0.005, 0.001, momentum of 0.95, 0.90, 0.85, and weight decay of 0.0005, 0.0010, 0.0015, and graphed the train loss, test loss, and test accuracy to find the best value for each parameter.

<insert graphs>


## Results 
To reach our final model, we experimented with various forms of data augmentation methods, expecting that it would allow the dataset to be trained on a wider breadth of imaging. However, adding data augmentation did not change the accuracy of the model much. One of the contributors to this could be that MRI scans have very little variation and data augmentation wouldn’t increase the variability of the input images by much. 

We didn’t want to apply noise since MRI technicians are trained to create good MRI scans. Thus, we applied transformations such as horizontal and vertical flips which wouldn’t affect black and white MRI scans that much.

We also found that changes to model depth or breadth also weren’t able to increase accuracy past the 97% threshold. After determining on a fairly simple model structure using three convolutional layers, we perform an experiment to find the best set of hyperparameters. 

### Final model:
<insert>

We analyzed accuracy using a binary correct or incorrect count and used a binary cross entropy loss with a sigmoid layer for stability. 

The findings suggest that deep learning models can accurately detect brain tumors and can be a valuable tool in clinical practice. However, an accuracy of 97%, although close to 100%, is nowhere near high enough for our model to be employed in a real world setting.

Future improvements on the model could include converting MRI images to grayscale or including more patient information inputs to the model such as age and health habits.





## Examples 
- images/text/live demo, anything to show off your work (note, demos get some extra credit in the rubric)

## Video 
- a 2-3 minute long video where you explain your project and the above information

