# CNN

Execute various "pip install ....." commands to install required packages.

First, execute generateNumpyArrayfromImage.py to create numpy file from image files. 

CATEGORIES=['1','2'] represents the folders including .jpg images. In that example, we have two classes.

Second, execute convolution.py to produce accuracy results.

The lines below save label and feature data sets.
np.save("/truba/home/muozturk/train.npy",X)
np.save("/truba/home/muozturk/trainLabel.npy",y)
