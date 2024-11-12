# reyesmatte_2024_imaging_paper
Repository to access the different notebooks and information of the paper ""

Here you'll find the information of the different environments used for specific tasks (data plotting, restoration, segmentation)

The `.yml` files are directly exported from conda environments used in the study, and are found in the respective folder of the task they're used for.

We also list different folders, each with their own contents:

## CARE 

- `care.yml`: conda environment file with the software specifications when training and segmenting
- `CARE_FM2FM.zip`: Compressed folder containing the CARE model. The `Predict.ipynb` notebook shows how to load it. Requires tensorflow
 
### The following notebooks are adapted from the official [CSBDeep repository](https://github.com/CSBDeep/CSBDeep)
- `Preparation.ipynb`: transforming the data into patches and exporting them into .npz files
- `Train.ipynb`: model training with the same parameters of the one used in the paper. GPU **very** necessary
- `Predict.ipynb`: notebook showing how to load a trained model and use it on new data

## Omnipose

- `omnipose.yml`: conda environment file with the software specifications when training and segmenting
- `Omnipose_CLI.txt`: the command used to train the Omnipose segmentation model, including specifications of hardware
- `Omnipose_segmentation.ipynb`: Jupyter notebook showing the parameters used for segmenting test and case images
- `Omnipose_metrics.ipynb`: Jupyter notebook showing the calculation of hte segmentation metrics in Figure X of the paper, as well as showing some of the testing data.
- `FMDeconvolved_2D_Omnipose`: Trained omnipose model. Can be loaded either in the GUI or through a notebook. Requires Pytorch


### References
- Cutler, K. J., Stringer, C., Lo, T. W., Rappez, L., Stroustrup, N., Brook Peterson, S., … & Mougous, J. D. (2022). Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation. _Nature methods_, 19(11), 1438-1448.
- Weigert, M., Schmidt, U., Boothe, T., Müller, A., Dibrov, A., Jain, A., ... & Myers, E. W. (2018). Content-aware image restoration: pushing the limits of fluorescence microscopy. _Nature methods_, 15(12), 1090-1097.
