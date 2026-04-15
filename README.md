# MEDUSSA (MEmbrane DeconvolUtion and Segmentation for Size Analyses)
![Schematic of the MEDUSSA workflow. The “logo” is a jellyfish (in Spanish: medusa) whose head has a gray gradient representing the distance of each pixel to its border. The tentacles of the jellyfish are two types: very thin black tentacles (which represent cell skeletons) and thicker, white, chain-like filaments (which represent the model organism of this work, Bacillus subtilis). Below the jellyfish there’s three images: First, a “Membrane image”, which is gray contours of cells on a black background. The cells are forming chains, and some of the fluorescence gets inside the contours. Next, a “Predicted deconvolution”, which is the same membranes as before, but now with sharper contours, and almost no fluorescence getting inside. Finally, an “Instance segmentation” where each individual cell has been colored to represent they have individual IDs. They kinda look like long cylindrical jellybeans. Between the “Membrane image” and “Predicted deconvolution”, as well as between the “Predicted deconvolution” and “Instance segmentation”, there’s a right-pointing arrow with a U-Net sketch, signifying how we use that architecture to predict the images. To the right of all, the outputs you’ll get with our pipeline are plots under the title “Size statistics and error propagation”. First, a scatterplot of blue points showing the relationship between two cell size measurements. To its right, two histograms on top of each other (one orange, one blue), which represents the error propagation aspect of our work. Above this title, two images which have the same cells of the “Instance segmentation”, but colored differently. As the title “Color cells by their cell size measurements” says, we can also color the instance segmentation by the calculated size metrics.](https://github.com/OReyesMatte/MEDUSSA/blob/main/workflow_full.png)

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

We strongly recommend using conda or any other environment manager to prevent compatibility issues between libraries. For this, we recommend using a environment manager like [miniforge](https://github.com/conda-forge/miniforge). Follow the installation instructions for your system. Another benefit is that having an environment with Omnipose allows running the segmentation on FIJI using the [BIOP wrappers](https://github.com/BIOP/ijl-utilities-wrappers) (follow the link for explanations on how to install and use!). In this case, we provide an easy installer for using the Omnipose fine-tuned models. Please refer to the [Cellpose](https://cellpose.readthedocs.io/en/latest/) and [microSAM](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html) documentations for instructions on their installation instructions. 


Next, find and open a terminal window, and run the following command:

```
conda create -n medussa_env -c conda-forge python=3.12 numpy -y && conda activate medussa_env
```

If you are only interested in the measuring functions, the easiest is to clone the repo
```
git clone https://github.com/OReyesMatte/MEDUSSA.git
```

Or download the source code archive
```
curl -L -O https://github.com/OReyesMatte/MEDUSSA/archive/master.zip
```

Then unzip it, either manually or with `tar`

```
tar -xf master.zip
```

Then go to the directory. 
```
cd MEDUSSA
```

If you downloaded it with `curl` and then unzipped it, the directory name will be "MEDUSSA-main'
```
cd MEDUSSA-main
```

In your terminal, with the `medussa_env` environment, run

```
pip install -r requirements.txt
```

To then verify the installation, open the python interpreter:

```
python
```

Then, run:
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

#### Windows details

If instead you are using windows, the installation will be trickier, and you'll have to modify the `peakdetect` file manually. Install with `pip` the segmentation and restoration libraries:
```
conda install python=3.12 -c conda-forge -y

pip install omnipose csbdeep pyarrow seaborn natsort
```
Next, find the `peakdetect.py` file. For example, in MacOS the path would be `miniforge3/envs/medussa_env/lib/python3.12/site-packages/peakdetect/peakdetect.py`. 
Then on the file change:

```
from scipy import fft, ifft
```

to
```
from scipy.fft import ifft
```

With this you'll make the fix and then can proceed

### Check the install

To then test the installation, run `omnipose` on your terminal. It will ask you to install the PyQt6 dependencies, type `y` and press enter to continue the installation. After that, the Omnipose GUI should open.
![Terminal screenshot, black background with white letters asking for the installation of GUI dependencies](https://github.com/OReyesMatte/MEDUSSA/blob/main/omnipose_installation.png)

Congrats! You successfully installed the necessary MEDUSSA libraries! You can run the `MEDUSSA_example.pynb` notebook to see the pipeline in action!

## Using MEDUSSA on your own data

Each aspect of the MEDUSSA pipeline (deconvolution prediction, segmentation, measurement) can be run independently from each other, so it can be integrated into existing pipelines. Each part indicates at the beginning what you need for each step. MEDUSSA has been tried and tested for widefield fluorescence microscopy, so we can't guarantee the performance on other fluorescence microscopy modalities (i.e., Confocal, Structured Illumination Microscopy, Superresolution Microscopy).

### Model training

If you wish to re-train the models, you'll need access to a computer with a GPU or access to HPC infrastructure due to resource demands of model training. Alternatively, you can access just the training images and use them in your own models, provided you give proper credit :-)

### Deconvolution prediction

**What you need**: CARE library installed, non-deconvolved fluorescent membrane images or cytoplasmic fluorescence images.

#### Prediction of 2D deconvolved membranes

Using either FM2FM, FP2FM or FM2FM-HiSNR.

```
from csbdeep.models import CARE
from csbdeep.utils import normalize
from tifffile import imread,imwrite
import numpy as np

model_dir = '/Users/reyesmatte/models/CARE/' 

model_2D = CARE(config=None, name=$MODEL_NAME$, basedir=model_dir)

img_2D = imread($IMAGE_PATH$)

prediction_2D = model_2D.predict(img=normalize(img_2D),axes='YX',n_tiles=(4,4))

imwrite($2D_OUTPUT_PATH$,prediction_2D)

### To predict on a 3D image:

img_3D = imread($3DIMAGE_PATH$)

prediction_3D = np.array([model_2D.predict(img=normalize(z),axes='YX',n_tiles=(4,4)) for z in img_3D] ## Make sure that the image shape is ZYX

imwrite($3D_OUTPUT_PATH$,prediction_3D)

### For the projection

img_3D = imread($3DIMAGE_PATH$)

projection = img_3D.mean(axis=0) ## Make sure that the image shape is ZYX)

prediction_projection =  model_2D.predict(img=normalize(projection),axes='YX',n_tiles=(4,4))

imwrite($PROJECTION_OUTPUT_PATH$,prediction_3D)

```

### Segmentation

**What you need**: your favorite segmentation library installed, fluorescent membrane images (raw or deconvolved)

Omnipose and microSAM performed better on deconvolved images, with Omnipose also being able to segment elongated cells and microSAM allowing for interactive segmentation with the napari plugin. Cellpose3 and CellposeSAM performed better in raw images.

For examples on how to load specfic models in Jupyter notebooks, please refer to each models folder in the "Segmentation" directory. Alternatively, check the respective documentations of each software for instructions for use with GUIs (napari or their owns).

The models will output instance segmentation masks that can be used for size quantification downstreams or extracting other type of parameters (i.e., fluorescence)

### Cell size quantification

**What you need**: instance segmentation masks 

To quantify the cells, each mask in your image must have a unique, integer ID. Most segmentation software generate these automatically. If not, ensure that you can get this type of mask. 

The 'MEDUSSA.measure' functions allow for the measurement of cells directly from a specific image or also from a list of images. We recommend the `SizeDataFrame` function, as this will provide a table with all the measured cells. 

The functions are optimized for rod-shaped (or pill-shaped) bacteria. Some functions can be adapted for measuring, for example, crescent shaped bacteria. However, it's not suitable for cells with irregular morphologies, and goes beyond what you'd need for measurements of spherical bacteria.

```
import numpy as np
import pandas as pd

from MEDUSSA.utils import BorderRemoval
from MEDUSSA.measure import SizeDataFrame

from glob import glob

masks_path = $IMAGE_PATH$

### If you instead read the image first, change to "from_files = False"
single_df = SizeDataFrame(maskfilelist = [masks_path], from_files = True, return_skeleton_paths = False, pixsize = 33.02/512) ## Pixel size of a Leica Thunder microscope

### More conveniently, the function can work on a list of files

masks_list = sorted(glob($FOLDER_WITH_MASKS/*$))
multi_df = SizeDataFrame(maskfilelist = [masks_list], from_files = True, return_skeleton_paths = False, pixsize = 33.02/512) ## Pixel size of a Leica Thunder microscope

```


## What else can you find here

### CARE 
The environment and notebooks to train the deconvolution prediction models outlined in the manuscript (refer to Figure 3 to see the results). Please refer to the [CSBDeep documentation](https://github.com/CSBDeep/CSBDeep) for installation instructions

- `care.yml`: conda environment file with the software specifications when training and segmenting

Trained models can be found in [zenodo](https://zenodo.org/records/18978187)

#### The following notebooks are adapted from the official [CSBDeep repository](https://github.com/CSBDeep/CSBDeep)
- `Preparation.ipynb`: transforming the data into patches and exporting them into .npz files for training
- `Train.ipynb`: model training with the same parameters of the one used in the paper. GPU **very** necessary
- `Predict.ipynb`: notebook showing how to load a trained model and use it on new data

Training and test images can be found at https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2353

### Segmentation
The environment files and commands for training the fine-tuned segmentation models outlined in the manuscript (refer to Figure 2 and Supplementary Figures 2 and 3 to see the results). 

#### Cellpose3
- `cellpose_CNN.yml`: conda environment file with the software specifications for using with GPUs. This environment was used for model training, and can also be used for segmentation.
- `cellpose3_CLI.txt`: the command used to train the Cellpose3 segmentation model, including specifications of hardware. GPU **very** necessary
- `Cellpose3_segmentation.ipynb`: Jupyter notebook exemplifying how to load a custom model and run it

#### CellposeSAM
- `cellpose.yml`: conda environment file with the software specifications for using with GPUs. This environment was used for model training, and can also be used for segmentation.
- `cellposeSAM_CLI.txt`: the command used to train the CellposeSAM segmentation model, including specifications of hardware. GPU **very** necessary
- `CellposeSAM_segmentation.ipynb`: Jupyter notebook exemplifying how to load a custom model and run it

#### microSAM
- `microsam.yml`: conda environment file with the software specifications for using with GPUs. This environment was used for model training, and can also be used for segmentation.
- `microSAM_train.ipynb`: Jupyter notebook exemplifying how to train a custom instance segmentation model. GPU **very** necessary
- `microSAM_automaticSegmentation.ipynb`: Jupyter notebook exemplifying how to load a custom model and run it

#### Omnipose
- `omnipose.yml`: conda environment file with the software specifications for using with GPUs. This environment was used for model training, and can also be used for segmentation.
- `Omnipose_CLI.txt`: the command used to train the Omnipose segmentation model, including specifications of hardware. GPU **very** necessary
- `Omnipose_segmentation.ipynb`: Jupyter notebook exemplifying how to load a custom model and run it

Fine-tuned segmentation models for raw and deconvolved images can be found in [zenodo](https://zenodo.org/records/18978187)
Training and test images and masks can be found at https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2350

### Figures

In the "Figures" folder, you'll find the Jupyter notebooks for reproducing graphs and plots in the different figures of the paper. If image data needs to be downloaded (i.e., for benchmarking), there will be a `wget` cell in the corresponding notebook. Work is ongoing to make it so everything that's reading and downloading dta is directly incorporated in the notebook and minimizes external downloads.

### Raw data

Raw data `.csv` files for the plots in all the figures 



### References

- [Main article](https://www.biorxiv.org/content/10.1101/2025.10.26.684635v1) Reyes-Matte, M., Fortmann-Grote, C., Gericke, B., Hüttman, N., Ojkic, N., & Lopez-Garrido, J. (2025). Deep-learning deconvolution and segmentation of fluorescent membranes for high precision bacterial cell size profiling
- [CARE](https://www.nature.com/articles/s41592-018-0216-7) Weigert, M., Schmidt, U., Boothe, T., Müller, A., Dibrov, A., Jain, A., ... & Myers, E. W. (2018). Content-aware image restoration: pushing the limits of fluorescence microscopy. _Nature methods_, 15(12), 1090-1097.
- [Omnipose](https://www.nature.com/articles/s41592-022-01639-4) Cutler, K. J., Stringer, C., Lo, T. W., Rappez, L., Stroustrup, N., Brook Peterson, S., … & Mougous, J. D. (2022). Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation. _Nature methods_, 19(11), 1438-1448.
- [Cellpose3](https://www.nature.com/articles/s41592-025-02595-5) Stringer, C., & Pachitariu, M. (2025). Cellpose3: one-click image restoration for improved cellular segmentation. Nature methods, 22(3), 592-599.
- [microSAM](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1.abstract) Pachitariu, M., Rariden, M., & Stringer, C. (2025). Cellpose-SAM: superhuman generalization for cellular segmentation. BioRxiv, 2025-04.
- [Cellpose-SAM](https://www.nature.com/articles/s41592-024-02580-4) Archit, A., Freckmann, L., Nair, S., Khalid, N., Hilt, P., Rajashekar, V., ... & Pape, C. (2025). Segment anything for microscopy. Nature methods, 22(3), 579-591.

