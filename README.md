# MEDUSSA (MEmbrane DeconvolUtion and Segmentation for Size Analyses)
Repository to access the different notebooks and information of the paper "High-precision bacterial size profiling: dissecting mechanisms of cell-size diversity in *Priestia megaterium*"

Here you'll find the information of the different environments used for specific tasks (data plotting, restoration, segmentation)

The `.yml` files are directly exported from conda environments used in the study,which were used in a HPC computing cluster, and are found in the respective folder of the task they're used for.

## MEDUSSA
A set of functions to measure rod-shaped cells from segmentation masks, estimate parameters to transform the data to account for segmentation error propagation, and sample data distributions. 

- `measure.py`: functions that calculate cell size measurement images of segmentation masks in cell size measurements: Length, Width, Surface Area, and Volume
- `transform.py`: functions that allow to transform the obtained metrics either by sampling parameters from a linear relationship to calculating confidence intervals
- `utils.py`: functions for changing segmentation labels, removing truncated edge masks, and calculating distribution intersections
- `requirements.txt`: minimum software requirements to run the functions in both `measure.py` and `transform.py` 
- `MEDUSSA_example.ipynb`: example notebook on how to load `MEDUSSA` and run the whole pipeline of deconvolution, segmentation, and measurement
  
## Installing MEDUSSA
If you are only interested in the measuring functions, you can install them with `pip install medussa`.

The installation of all the libraries to run the full MEDUSSA pipeline (Deconvolution, Segmentation, Measurement) can be tricky mainly because of two factors:

- CARE runs on TensorFlow and Omnipose on PyTorch, and existing environments can make clashes between the two softwares
- One of the libraries Omnipose uses, peakdetect, has not been mantained for many years, and one of the functions it calls requires very old versions of SciPy to keep consistent function calls

Installation of both _the necessary TensorFlow and PyTorch_ to run both CARE and Omnipose can be done in a fresh environment, for which we provide the instructions below.

If you prefer to keep everything in a pure PyTorch environment, we're working on versions of the CARE models using the [CAREamics](https://github.com/CAREamics/careamics) framework.

### Installation (macOS and Linux)
For this, we recommend using a environment manager like [miniforge](https://github.com/conda-forge/miniforge). Follow the installation instructions for your system. Another benefit is that having an environment with Omnipose allows running the segmentation on FIJI using the [BIOP wrappers](https://github.com/BIOP/ijl-utilities-wrappers) (follow the link for explanations on how to install and use!).

Next, find and open a terminal window, and run the following command:

`conda create -n medussa_env -y && conda activate medussa_env`

This will create and put you in a fresh environment so you can install the necessary libraries.

Now, download the `medussa_install` file that corresponds to your operating system. The distinction is because newer versions of TensorFlow have separate install sites for different operating systems, so the distintction is necessary to allow for deconvolutions.

In the same terminal that you opened and in the `medussa_env` environment, run `sh medussa_install_macos` or `sh medussa_install_linux` according to your operating system. This will take a few minutes.

To then test the installation, run `omnipose` on your terminal. It will ask you to install the PyQt6 dependencies, type `y` and press enter to continue the installation. After that, the Omnipose GUI should open.

Congrats! You successfully installed the necessary MEDUSSA libraries! Time to process your images!


## CARE 
The environment and notebooks to train the deconvolution prediction models outlined in the manuscript (refer to Figure 3 to see the results). Please refer to the [CSBDeep documentation](https://github.com/CSBDeep/CSBDeep) for installation instructions

- `care.yml`: conda environment file with the software specifications when training and segmenting
 
### The following notebooks are adapted from the official [CSBDeep repository](https://github.com/CSBDeep/CSBDeep)
- `Preparation.ipynb`: transforming the data into patches and exporting them into .npz files for training
- `Train.ipynb`: model training with the same parameters of the one used in the paper. GPU **very** necessary
- `Predict.ipynb`: notebook showing how to load a trained model and use it on new data

Training images can be found in:

## Omnipose
The environment and command train the segmentation models outlined in the manuscript (refer to Figure 2 and Supplementary Figure 2 to see the results). Please refer to the [Omnipose documentation](https://omnipose.readthedocs.io/) for installation instructions

- `omnipose_GPU.yml`: conda environment file with the software specifications for using with GPUs. This environment was used for model training, and can also be used for segmentation.
- `Omnipose_CLI.txt`: the command used to train the Omnipose segmentation model, including specifications of hardware. GPU **very** necessary
- `Omnipose_segmentation.ipynb`: Jupyter notebook exemplifying how to load a custom model and running it

Images (training and testing images and masks) can be found in:

## Figure reproducibility

In the "Figures" folder, you'll find each element necessary to reproduce the figures from the paper, where each figure has a corresponding folder. Inside the folders, you can find:
- Jupyter notebooks for reproducing graphs and plots
- Data tables (in `.csv` or `.xlsx` format) that are either: generated by the corresponding notebook in the folder, or generated externally but read by the figure's notebook
- If image data needs to be downloaded (i.e., for benchmarking), there will be a `wget` cell in the corresponding notebook

### References

- [Main article]() Reyes-Matte, M., Fortmann-Grote, C., Gericke, B., Hüttman, N., Ojkic, N., & Lopez-Garrido, J. (2025). Deep-learning-based deconvolution and segmentation of fluorescent membranes for high precision bacterial cell size profiling
- [CARE](https://www.nature.com/articles/s41592-018-0216-7) Weigert, M., Schmidt, U., Boothe, T., Müller, A., Dibrov, A., Jain, A., ... & Myers, E. W. (2018). Content-aware image restoration: pushing the limits of fluorescence microscopy. _Nature methods_, 15(12), 1090-1097.
- [Omnipose](https://www.nature.com/articles/s41592-022-01639-4) Cutler, K. J., Stringer, C., Lo, T. W., Rappez, L., Stroustrup, N., Brook Peterson, S., … & Mougous, J. D. (2022). Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation. _Nature methods_, 19(11), 1438-1448.

