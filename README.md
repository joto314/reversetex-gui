# reversetex-gui
A small TkInter Application which is able to (somewhat) identify LaTeX characters written on it using already trained Convolutional Neural Networks.

Due to GitHubs filesize limitation and potential copyright issues models and training data are *not* included.

## Files:
* createvars.py reads and processes the hasyv2 dataset
* NN.py trains three different Convolutional Neural Networks on the hasyv2 dataset
* gui.py provides a GUI to test the trained models
