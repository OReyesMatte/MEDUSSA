from scipy.stats import gaussian_kde
import numpy as np

def Relabeler(ground_truth:np.array,segmentation:np.array)->np.array:

    assert ground_truth.shape==segmentation.shape, "Image dimensions don't match!"
    
    binary = ground_truth>0

    new_labels = segmentation*binary

    kept_labels = np.isin(segmentation,np.unique(new_labels)[1:])

    return kept_labels*segmentation

def BorderRemoval(mask:np.array):

    def BorderElements(array:np.array, width:int): 
    
        n = array.shape[0]
        r = np.minimum(np.arange(n)[::-1], np.arange(n))
    
        a =  array[np.minimum(r[:,None],r)<width]

        return a[a.nonzero()]

    
    borderIDs = BorderElements(mask,2)

    if len(borderIDs) == 0:
        return mask

    else:
        
        CopyArray = np.copy(mask)

        for ID in borderIDs:

            Negative_mask = (mask != ID)
        
            CopyArray *= Negative_mask
    
        return CopyArray


def IntersectionKDE(x0,x1):

    kde0 = gaussian_kde(x0, bw_method='scott')
    kde1 = gaussian_kde(x1, bw_method='scott')

    xmin = min(x0.min(), x1.min())
    xmax = max(x0.max(), x1.max())
    
    #dx = 0.1 * (xmax - xmin) # add a 20% margin, as the kde is wider than the data
    #xmin -= 0.1
    #xmax += 0.1
    
    x = np.linspace(xmin, xmax, 1000)
    kde0_x = kde0(x)
    kde1_x = kde1(x)
    inters_x = np.minimum(kde0_x, kde1_x)
    
    return x,kde0_x,kde1_x,inters_x

