import cv2
import copy
import glob
import random
import numpy as np  # for transformation
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
#ez


from torch.utils.data import Dataset, DataLoader, dataloader
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def flatten(t):
    return [item for sublist in t for item in sublist]

  #######################################################
  #               Define Dataset Class
  #######################################################

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=False, class_to_idx=None):
        self.image_paths = image_paths
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = image_filepath.split('\\')[-2]
        label = self.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 50)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            # Defining another 2D convolution layer
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            # Defining another 2D convolution layer
            nn.Conv2d(16, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(4608, 400),
            nn.Linear(400, 84),
            nn.Linear(84, 6)
        )


    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.dropout1(x)
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(x.size(0), -1)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def load_datasets(datasets, batch_size=64, num_workers=2):
    train_dataset = datasets[0]
    valid_dataset = datasets[1]
    test_dataset = datasets[2]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return (train_loader, valid_loader, test_loader)



def create_datasets(train_data_path = 'images\\train', test_data_path = 'images\\test'):
    #######################################################
    #               Define Transforms
    #######################################################

    # To define an augmentation pipeline, you need to create an instance of the Compose class.
    # As an argument to the Compose class, you need to pass a list of augmentations you want to apply.
    # A call to Compose will return a transform function that will perform image augmentation.
    # (https://albumentations.ai/docs/getting_started/image_augmentation/)

    train_transforms = A.Compose(
        [
            A.Resize(128,128),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.25),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15,b_shift_limit=15, p=0.25),
            A.RandomBrightnessContrast(p=0.1),
            A.MultiplicativeNoise(multiplier=[0.5, 2], per_channel=True, p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.25),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.25),
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.Resize(128,128),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    ####################################################
    #       Create Train, Valid and Test sets
    ####################################################

    train_image_paths = []  # to store image paths in list
    classes = []  # to store class values
    print(glob.glob(train_data_path))

    # 1.
    # get all the paths from train_data_path and append image paths and class to to respective lists
    for data_path in glob.glob(train_data_path + '/*'):
        classes.append(data_path.split('\\')[-1])
        train_image_paths.append(glob.glob(data_path + '/*'))
    train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)

    print('Train_image_path example: ', train_image_paths[0])
    print('Class example: ', classes[0])

    train_image_paths, valid_image_paths = train_image_paths[:int(
        0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):]

    # 3.
    # create the test_image_paths
    test_image_paths = []
    for data_path in glob.glob(test_data_path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*'))

    test_image_paths = list(flatten(test_image_paths))

    print("Train size: {}\nValid size: {}\nTest size: {}".format(
        len(train_image_paths), len(valid_image_paths), len(test_image_paths)))

    #######################################################
    #      Create dictionary for class indexes
    #######################################################

    idx_to_class = {i: j for i, j in enumerate(classes)}
    class_to_idx = {value: key for key, value in idx_to_class.items()}

    #######################################################
    #                  Create Dataset
    #######################################################

    train_dataset = CustomDataset(train_image_paths, train_transforms, class_to_idx=class_to_idx)
    valid_dataset = CustomDataset(valid_image_paths, test_transforms, class_to_idx=class_to_idx)  # test transforms are applied
    test_dataset = CustomDataset(test_image_paths, test_transforms, class_to_idx=class_to_idx)

    #print('The shape of tensor for 50th image in train dataset: ',
          #train_dataset[0][0].shape)
    #print('The label for 50th image in train dataset: ', train_dataset[0][0])
    return (classes, train_dataset, valid_dataset, test_dataset)

def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

def visualize_augmentations(dataset, idx=0, samples=10, cols=5, random_img=False):

        dataset = copy.deepcopy(dataset)
        # we remove the normalize and tensor conversion from our augmentation pipeline
        dataset.transform = A.Compose(
            [t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
        rows = samples // cols

        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
        for i in range(samples):
            if random_img:
                idx = np.random.randint(1, len(train_image_paths))
            image, lab = dataset[idx]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
            ax.ravel()[i].set_title(idx_to_class[lab])
        plt.tight_layout(pad=1)
        plt.show()

    #visualize_augmentations(train_dataset,np.random.randint(1,len(train_image_paths)), random_img = True)


def train_new_model(classes, dataset, num_epochs, name='saved_model.pth'):
    train_loader = dataset[0]
    valid_loader = dataset[1]
    test_loader = dataset[2]

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))

    net = Net()
    min_valid_loss = np.inf

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.1)
    #optimizer = optim.Adadelta(net.parameters(), lr=0.1)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        net.train(True)
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if epoch % 5 == 0: 
            net.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for data, labels in valid_loader:
                    target = net(data)
                # Find the Loss
                    loss = criterion(target,labels)
                # Calculate Loss
                    valid_loss += loss.item()
        
                print(f'Epoch {epoch} \t\t Training Loss: {running_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
            
                if min_valid_loss > valid_loss:
                    print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                    min_valid_loss = valid_loss
                    torch.save(net.state_dict(), 'saved_model3.pth')
        else:
            print(f'Epoch {epoch} \t\t Training Loss: {running_loss / len(train_loader)}')

    print('Finished Training')

    dataiter = iter(test_loader)
    images,labels = dataiter.next()

    # print images
    #imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' %
                                    classes[labels[j]] for j in range(4)))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 1250 test images: %d %%' % (
        100 * correct / total))

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))

def load_run(test_dataset, classes, path):
    net = Net()
    net.load_state_dict(torch.load(path))
    net.eval()

    dataiter = iter(test_dataset)
    images,labels = dataiter.next()

    # print images
    #imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' %
                                    classes[labels[j]] for j in range(4)))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataset:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 1250 test images: %d %%' % (
        100 * correct / total))

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_dataset:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))



if __name__ == '__main__':
    classes, *dataset_raw = create_datasets()
    dataloaders = load_datasets(dataset_raw)
    train_new_model(classes, dataloaders, 26, name='saved_model4.pth')
    #load_run(dataloaders[2], classes,'saved_model3.pth')
