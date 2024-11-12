# reyesmatte_2024_imaging_paper
Repository to access the different notebooks and information of the paper ""

Here you'll find the information of the different environments used for specific tasks (data plotting, restoration, segmentation)

The `.yml` files are directly exported from conda environments used in the study, and are found in the respective folder of the task they're used for.

We also list different folders, each with their own contents:

## Omnipose

- `omnipose.yml`: conda environment file with the software specifications when training and segmenting
- `Omnipose_CLI.txt`: the command used to train the [Omnipose](https://omnipose.readthedocs.io/) segmentation model, including specifications of hardware
- `Omnipose_segmentation.ipynb`: Jupyter notebook showing the parameters used for segmenting test and case images
- `Omnipose_metrics.ipynb`: Jupyter notebook showing the calculation of hte segmentation metrics in Figure X of the paper, as well as showing some of the testing data.
