from scipy.stats import gaussian_kde
import numpy as np

def Relabeler(template_masks:np.array,input_masks:np.array)->np.array:

    """Taking two related instance segmentation image (i.e., ground-truth labels and benchmarking image), ensures only the same cells are compared.
    This should NOT be used for calculating accuracy metrics. It could be used for comparing downstream analyses on masks.

    Args:
        - template_masks(np.array): the masks that are used as a "map" to curate the input image
        - input_masks(np.array): the masks that will be curated according to the template image

    Returns:
        - an array of all the kept masks from the input image
    """
    
    assert template_masks.shape==input_masks.shape, "Image dimensions don't match!"
    
    binary = template_masks>0

    new_labels = input_masks*binary

    kept_labels = np.isin(input_masks,np.unique(new_labels)[1:])

    return kept_labels*input_masks

def BorderElements(masks:np.array, border:int): 

    """Identify segmentation masks that are on the border on the image and likely truncated and remove them from segmentation image

    Args:
        masks(np.array): instance segmentation masks
        border(int): how many pixels are to be considered part of the border

    Returns:
        an array with the labels of objects on the borders
    """

    n = masks.shape[0]
    r = np.minimum(np.arange(n)[::-1], np.arange(n))

    a =  masks[np.minimum(r[:,None],r) < border]

    return a[a.nonzero()]

def BorderRemoval(masks:np.array,border:int=2)->np.array:

    """Remove the segmentation masks that are on the edge of the image. If a mask is truncated, we don't know how much mask is missing, therefore making size calculations innacurate
    
     Args:
        masks(np.array): instance segmentation masks

    Returns: 
        If no cells are found in the border, returns the same image
        Else, it returns the mask image with the cells removed, keeping the original labels of the remaining cells
    """

    borderIDs = BorderElements(masks=masks,border=border)

    if len(borderIDs) == 0:
        return masks

    else:
        
        CopyArray = np.copy(masks)

        for ID in borderIDs:

            Negative_mask = (masks != ID)
        
            CopyArray *= Negative_mask
    
        return CopyArray


def IntersectionKDE(x0:np.array,x1:np.array):

    """Calculate the Kernel Density Estimate (KDE) for two data distributions and compute their intersection.
    Please refer to the  gaussian_kde documentation in https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html from SciPy for more details on the estimation
    
    Args:
        x0(np.array): an array of continuous values of population/group/condition/strain 0
        x1(np.array): an array of continuous values of population/group/condition/strain 1

    Returns:
        new_x: the new "x" array that works common for both KDEs
        kde_x0: the KDE results for input array x0
        kde_x1: the KDE results for input array x1
        intersection_x01: the intersection of both KDEs
    """

    kde0 = gaussian_kde(x0, bw_method='scott')
    kde1 = gaussian_kde(x1, bw_method='scott')

    xmin = min(x0.min(), x1.min())
    xmax = max(x0.max(), x1.max())
    
    xmin -= 0.1
    xmax += 0.1
    
    new_x = np.linspace(xmin, xmax, 1000)
    kde_x0 = kde0(new_x)
    kde_x1 = kde1(new_x)
    intersection_x01 = np.minimum(kde_x0, kde_x1)
    
    return new_x,kde_x0,kde_x1,intersection_x01

def CellColorer(masks:np.array,labels,values)->np.array:

    """Color segmentation masks in an image according to a quantitative metric.
    
    Args:
        masks(np.array): instance segmentation masks
        labels(): labels of the cells that have been measured
        values(): values to color the cells

    Returns:
        value_image(np.array): segmentation masks colored by chosen metric
    """

    value_image = np.zeros(masks.shape)

    for label,value in zip(labels,values):

         value_image += (masks==label)*value

    return value_image