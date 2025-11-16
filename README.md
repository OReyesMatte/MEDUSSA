# MEDUSSA (MEmbrane DeconvolUtion and Segmentation for Size Analyses)
![Schematic of the MEDUSSA workflow. The “logo” is a jellyfish (in Spanish: medusa) whose head has a gray gradient representing the distance of each pixel to its border. The tentacles of the jellyfish are two types: very thin black tentacles (which represent cell skeletons) and thicker, white, chain-like filaments (which represent the model organism of this work, Bacillus subtilis). Below the jellyfish there’s three images: First, a “Membrane image”, which is gray contours of cells on a black background. The cells are forming chains, and some of the fluorescence gets inside the contours. Next, a “Predicted deconvolution”, which is the same membranes as before, but now with sharper contours, and almost no fluorescence getting inside. Finally, an “Instance segmentation” where each individual cell has been colored to represent they have individual IDs. They kinda look like long cylindrical jellybeans. Between the “Membrane image” and “Predicted deconvolution”, as well as between the “Predicted deconvolution” and “Instance segmentation”, there’s a right-pointing arrow with a U-Net sketch, signifying how we use that architecture to predict the images. To the right of all, the outputs you’ll get with our pipeline are plots under the title “Size statistics and error propagation”. First, a scatterplot of blue points showing the relationship between two cell size measurements. To its right, two histograms on top of each other (one orange, one blue), which represents the error propagation aspect of our work. Above this title, two images which have the same cells of the “Instance segmentation”, but colored differently. As the title “Color cells by their cell size measurements” says, we can also color the instance segmentation by the calculated size metrics.](https://github.com/OReyesMatte/MEDUSSA/blob/main/workflow.png)

Repository to access the different notebooks and information of the paper [Deep-learning deconvolution and segmentation of fluorescent membranes for high precision bacterial cell size profiling](https://www.biorxiv.org/content/10.1101/2025.10.26.684635v1)

Here you'll find the information of the different environments used for specific tasks (data plotting, restoration, segmentation)

We provide `.yml` files of the different environments used for training the models in the study, which were used in a HPC computing cluster, and are found in the respective folder of the task they're used for.

## MEDUSSA
A set of functions to measure rod-shaped cells from segmentation masks, estimate parameters to transform the data to account for segmentation error propagation, and sample data distributions. 

- `measure.py`: functions that calculate cell size measurement images of segmentation masks in cell size measurements: Length, Width, Surface Area, and Volume
- `transform.py`: functions that allow to transform the obtained metrics either by sampling parameters from a linear relationship to calculating confidence intervals
- `utils.py`: functions for changing segmentation labels, removing truncated edge masks, and calculating distribution intersections

The notebook `MEDUSSA_example.ipynb` shows an example on how to load `MEDUSSA` and run the whole pipeline of deconvolution, segmentation, and measurement
  
### Installation

We strongly recommend using conda or any other environment manager to prevent compatibility issues between libraries. For this, we recommend using a environment manager like [miniforge](https://github.com/conda-forge/miniforge). Follow the installation instructions for your system. Another benefit is that having an environment with Omnipose allows running the segmentation on FIJI using the [BIOP wrappers](https://github.com/BIOP/ijl-utilities-wrappers) (follow the link for explanations on how to install and use!).


Next, find and open a terminal window, and run the following command:

```
conda create -n medussa_env -c conda-forge python=3.11 numpy -y && conda activate medussa_env
```

If you are only interested in the measuring functions, the easiest is to clone the repo
```
git clone https://github.com/OReyesMatte/MEDUSSA/.git
```

Or download the source code archive
```
wget https://github.com/OReyesMatte/MEDUSSA/archive/master.zip
unzip master.zip -d MEDUSSA
```

Then go to the directory 
```
cd MEDUSSA
```

### MEDUSSA functions only using `pip`

In your terminal, with the `medussa_env` environment, run

```
pip install -r requirements.txt
```

Then, on your terminal, run `python`, which will open the Python interpreter, there runt:
```
from MEDUSSA.utils import InstallCheck
InstallCheck()
```
If the output message is `"All the base MEDUSSA functions can be used!"`, congrats! You can start measuring your segmented cells! If not, the function will specify which libraries are not installed. All of them can be installed with `pip`.

### All the functions used in the paper

The installation of all the libraries to run the full MEDUSSA pipeline (Deconvolution, Segmentation, Measurement) can be tricky mainly because of two factors:

- CARE runs on TensorFlow and Omnipose on PyTorch, and existing environments can make clashes between the two libraries
- One of the libraries Omnipose uses, `peakdetect`, has not been mantained for many years, and one of the functions it calls requires very old versions of SciPy to keep consistent function calls

Installation of both _the necessary TensorFlow and PyTorch_ to run both CARE and Omnipose can be done in a fresh environment (like the `medussa_env` created in the previous step), for which we provide the instructions below.

If you prefer to keep everything in a pure PyTorch environment, we're working on versions of the CARE models using the [CAREamics](https://github.com/CAREamics/careamics) framework.

In the same terminal that you opened and in the `medussa_env` environment, run `sh medussa_install_macos` or `sh medussa_install_linux` according to your operating system. This will take a few minutes.

To then test the installation, run `omnipose` on your terminal. It will ask you to install the PyQt6 dependencies, type `y` and press enter to continue the installation. After that, the Omnipose GUI should open.
![Terminal screenshot, black background with white letters asking for the installation of GUI dependencies](https://github.com/OReyesMatte/MEDUSSA/blob/main/omnipose_installation.png)


Congrats! You successfully installed the necessary MEDUSSA libraries! You can run the `MEDUSSA_example.pynb` notebook to see the pipeline in action!


## Model training

If you wish to re-train the models, you'll need access to a computer with a GPU or access to HPC infrastructure due to resource demands of model training. Alternatively, you can access just the training images and use them in your own models, provided you give proper credit :-)

### CARE 
The environment and notebooks to train the deconvolution prediction models outlined in the manuscript (refer to Figure 3 to see the results). Please refer to the [CSBDeep documentation](https://github.com/CSBDeep/CSBDeep) for installation instructions

- `care.yml`: conda environment file with the software specifications when training and segmenting
- `FM2FM.zip`: the deconvolved membrane prediction model from another fluorescence membrane image.
- `FM2FM.zip`: the deconvolved membrane prediction model from a cytoplasmic fluorescence.
 
#### The following notebooks are adapted from the official [CSBDeep repository](https://github.com/CSBDeep/CSBDeep)
- `Preparation.ipynb`: transforming the data into patches and exporting them into .npz files for training
- `Train.ipynb`: model training with the same parameters of the one used in the paper. GPU **very** necessary
- `Predict.ipynb`: notebook showing how to load a trained model and use it on new data

Training and test images can be found at https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2353

### Omnipose
The environment and command train the segmentation models outlined in the manuscript (refer to Figure 2 and Supplementary Figure 2 to see the results). Please refer to the [Omnipose documentation](https://omnipose.readthedocs.io/) for installation instructions

- `omnipose_GPU.yml`: conda environment file with the software specifications for using with GPUs. This environment was used for model training, and can also be used for segmentation.
- `Omnipose_CLI.txt`: the command used to train the Omnipose segmentation model, including specifications of hardware. GPU **very** necessary
- `Omnipose_segmentation.ipynb`: Jupyter notebook exemplifying how to load a custom model and running it
- `FMSeg`: the segmentation model for deconvolved membranes
- `RawFMSeg`: the segmentation model for non-deconvolved membranes

Training and test images and masks can be found at https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2350

## Figure reproducibility

In the "Figures" folder, you'll find each element necessary to reproduce the figures from the paper, where each figure has a corresponding folder. Inside the folders, you can find:
- Jupyter notebooks for reproducing graphs and plots
- Data tables (in `.csv` or `.xlsx` format) that are either: generated by the corresponding notebook in the folder, or generated externally but read by the figure's notebook
- If image data needs to be downloaded (i.e., for benchmarking), there will be a `wget` cell in the corresponding notebook

### References

- [Main article](https://www.biorxiv.org/content/10.1101/2025.10.26.684635v1) Reyes-Matte, M., Fortmann-Grote, C., Gericke, B., Hüttman, N., Ojkic, N., & Lopez-Garrido, J. (2025). Deep-learning deconvolution and segmentation of fluorescent membranes for high precision bacterial cell size profiling
- [CARE](https://www.nature.com/articles/s41592-018-0216-7) Weigert, M., Schmidt, U., Boothe, T., Müller, A., Dibrov, A., Jain, A., ... & Myers, E. W. (2018). Content-aware image restoration: pushing the limits of fluorescence microscopy. _Nature methods_, 15(12), 1090-1097.
- [Omnipose](https://www.nature.com/articles/s41592-022-01639-4) Cutler, K. J., Stringer, C., Lo, T. W., Rappez, L., Stroustrup, N., Brook Peterson, S., … & Mougous, J. D. (2022). Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation. _Nature methods_, 19(11), 1438-1448.

