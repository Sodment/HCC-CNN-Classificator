# HCC-CNN-Classificator
A simple convolutional neural network made using pytorch for Human-Computer-Communication course, the net scored 80% accuracy after training on the Intel dataset.
My object was to implement image classification in any method I desire.

# Data set
[Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification/version/2)

# How to run
Just run the image_recognizer.py

If you desire to train your own model please uncomment the following line (training may take around 10-20 minutes depending on your machine and if you enabled CUDA acceleration)

``` #train_new_model(classes, dataloaders, 31, name='saved_model.pth') ```

If you desire to use pretrained model please use the following line and comment the line above

``` load_run(dataloaders[2], classes,'saved_model.pth') ```

# CUDA
I dont have a CUDA GPU so I cant really use it, if you desire to speed up your training please refer to [pytroch documentation about CUDA](https://pytorch.org/docs/stable/notes/cuda.html)
