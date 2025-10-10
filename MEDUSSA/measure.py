# Array and data functions
import numpy as np
import pandas as pd

# Image processing 
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.measure import regionprops_table
from skimage.morphology import skeletonize, binary_erosion, binary_dilation
from skimage import io
from cv2 import findNonZero

# Math and statistics
import math

def euclidean_distance(p1:tuple,p2:tuple)->float:

    """Function to calculate the Euclidean distance between two points in a two-dimensional space

    Args:
        p1(tuple): coordinates (x,y) of the first point
        p2(tuple): coordinates (x,y) of the second point

    Returns: 
        float: distance value between the two points
    """

    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def compute_edges(points:list,threshold:float)->list:
    
    """Convert a list of spatial coordinates to an edge list, where each point is considered to be the edge of a specific point if it's below a certain distance.
    If it is only one point, then it won't go through, as it wouldn't be a rod-shaped cell

    Args:
        points(list): list of points coordinates (x,y)
        threshold(float): minimum distance for two points to be considered edges

    Returns: 
        edges: list of candidate edges for each point
    """
    
    edges = []

    if len(points) == 1:
        return print("Only one point, cannot be converted to graph")

    else:

        for i in range(len(points)):
            for j in range(i+1, len(points)):

                p1 = points[i]
                p2 = points[j]
                distance = euclidean_distance(p1,p2)
                if distance <= threshold:
                    edges.append((p1,p2))

        return edges

## Build graph as list
def build_graph(points:list,edges:list)->dict:

    """Convert a list of points and edges into a graph
    
    Args:
        points(list): list of points coordinates (x,y)
        threshold(float): minimum distance for two points to be considered edges

    Returns: 
        graph(dict): graph representation of the skeleton, where lists the edges of each point in the skeleton
    """

    graph = {point: [] for point in points}
    
    for p1,p2 in edges:
        graph[p1].append(p2)
        graph[p2].append(p1)

    return graph

def calculate_total_distance(path:list)->float:

    """Calculate the total distance as the seum of euclidean distances between two consecutive points
    
    Args:
        path(list): list of points in (x,y) coordinate

    Return:
        total_distance(float): total calculated distance as a single float number
    """

    total_distance = 0

    for i in range(len(path) - 1):
        total_distance += euclidean_distance(path[i],path[i+1])
    return total_distance

### Depth First Search
def DFS(graph, current, visited, path, longest_path, longest_distance):
    visited.add(current)
    path.append(current)

    ## Update path
    if len(path) > len(longest_path):
        longest_path[:] = path[:]
        longest_distance[0] = calculate_total_distance(path)

    for neighbor in graph[current]:
        if neighbor not in visited:
            DFS(graph, neighbor, visited, path, longest_path, longest_distance)

    visited.remove(current)
    path.pop

### Check for multible possible paths
def PathFinder(path:list,points:list,threshold:float)->list:

    """ It can happen that DFS computes multiple possible paths, leaving to wrong measurements. 
    For a candidate longest path, find if, indeed, is it just one path or multiple, as the latter can happen with branched skeletons
    If two successive points in the path are more than one threshold away (typically, the euclidean distance between two diagonal points), the path is assumed to split there into multiple
    If the path is too short (less than 10 pixels, or less than 0.65 µm of length), it's discarded.

    Args:
        path(list): 
        points(list): list of points coordinates (x,y)
        threshold(float): minimum distance for two points to be considered edges

    Returns: 
        paths(list): computed individual paths
    """

    paths = []

    j = 0
    
    for i in range(len(path)-1):

        if euclidean_distance(path[i],path[i+1]) > threshold:
            
            subpath = path[j:(i+1)]
            paths.append(subpath)
            j = i+1

    for i,p in enumerate(paths):

        if len(p) < 10:
            del paths[i]

    return paths

def WSV(skeleton_points:list,mask_distance:np.array,pixsize:float=33.02/512)->list:
    
    """Calculate cell width, surface area and volume across a single cell. Assuming a cell geometry of cylindrical body with rotational symmetry and hemispherical caps, where each point of the skeleton is treated as an individual "cylinder" of height 1 and a radius of the distance at that point.
    Width is computed as twice the distance from overlaying the cell skeleton in the distance transform of the cell
    Surface area is the sum of the lateral surface areas of each cylinder plus the external surface areas of each hemispherical cap
    Volume is the sum of the volume of each cylinder plus the volume of each hemispherical cap

    Args:
        skeleton_points(list): coordinates (x,y) of each point in the cell skeleton
        mask_distance(np.array): distance transform of the mask to be measured
        pixsize(float): spatial scale of pixels (info available in the microscope software or opening the image in, for example, FIJI). Necessary for converting measurements to standard units

    Returns: 
        Widths(float): width values across the cell skeleton
        Surface_area(float): surface area value of the cell
        Volume(float): volume value of the cell
"""
    pi=math.pi

    Widths = [mask_distance[coord[1],coord[0]]*pixsize for coord in skeleton_points]

    Widths_square = [w**2 for w in Widths]
    
    Surface_area = 2*pi*((Widths[0]**2) + (Widths[-1]**2) + pixsize*np.sum(Widths))

    Volume = pi*((Widths[0]**3)*2/3 +(Widths[-1]**3)*2/3 + pixsize*np.sum(Widths_square))

    Widths = np.array(Widths)*2
          
    return Widths, Surface_area, Volume

def SkeletonMeasure(skeleton:np.array,distances:np.array,pixsize:float=33.02/512,return_paths:bool=False):

    """Function to measure the cell size base on the skeleton overalid on the mask distance transform. The steps are:

        - Get the coordinates of the skeleton points
        - Compute edges and build graph based on the points
        - Calculate longest path of the skeleton
        - If the longest path has more points than the skeleton, compute the candidate paths
        - The candidate paths can be returned as a list in order to get per-position width values, or ignored and the median of metrics is returned
        - If the longest path is the same size, simply measure cell size and return the values
        - If there's only one point in the skeleton, the function simply returns 0 for the cell size values, as it's not a rod shape

    Args:
        skeleton(np.array): the binary skeleton of a single cell (computed with functions such as "skeletonize" or "medial_axis" from scikit-image)
        distances(np.arry): the distance transform of a single cell ((computed with a function such as "distance_transform_edt" from scipy)
        pixsize(float): the pixel sampling size. This information is available and intrinsic to each imaging system. Set as 1 to get the values in pixels.
        return_paths(bool): if you want to return all the candidate paths as a list

     Returns: 
        Length(float/list): length of the cell if there's only one candidate path, if return_paths=True is a list of possible lengths, else it returns the median length value of candidate paths
        Width(float/list): width values across the cell skeleton if there's only one candidate path or if return_paths=True, else it returns the median width value of candidate paths
        Surface_area(float/list): surface area of the cell if there's only one candidate path, if return_paths=True is a list of possible surface areas, else it returns the median surface area value of candidate paths
        Volume(float/list): volume of the cell if there's only one candidate path, if return_paths=True is a list of possible volume, else it returns the median volume value of candidate paths
    """

    try:

        threshold=1.5
        
        #### Get the coordinates of the skeleton points
        points = np.squeeze((findNonZero(np.uint8(skeleton))))
        points = [tuple(xy) for xy in points]

        if len(points) > 1:
            #### Compute edges and build graph
            edges = compute_edges(points=points,threshold=threshold)
            graph = build_graph(points=points,edges=edges)

            #### Calculate the longest path of the skeleton
            longest_path = []
            longest_distance = [0]
    
            for point in points:
                visited = set()
                DFS(graph,point,visited,[],longest_path,longest_distance)            

            if len(longest_path) > len(points):
                paths = PathFinder(path=longest_path,points=points,threshold=threshold)

                #### Get Width, Surface Area, and Volume

                newDistances = [calculate_total_distance(selected_paths)*pixsize for selected_paths in paths]
                
                M = [WSV(skeleton_points=selected_paths,mask_distance=distances) for selected_paths in paths]

                Ws = [arr[0] for arr in M]
                Ss = [arr[1] for arr in M]
                Vs = [arr[2] for arr in M]

                newDistances = [dist+w[0]/2+w[-1]/2 for dist,w in zip(newDistances,Ws)]
                
                if return_paths:

                    Length = newDistances
                    W = [w.mean() for w in Ws]
                    Surface_area = Ss
                    Volume = Vs

                    return Length,Width,Surface_area,Volume,paths
                
                else:

                    Length = np.median(newDistances)
                    Width = np.median([w.mean() for w in Ws])
                    Surface_area = np.median(Ss)
                    Volume = np.median(Vs)

                    return Length,Width,Surface_area,Volume,paths

            else:
                longest_path = longest_path
                
                #### Get Width, Surface Area, and Volume
                Width,Surface_area,Volume = WSV(skeleton_points=longest_path,mask_distance=distances)
                newDistance = calculate_total_distance(longest_path)
                
                #### For length, get the longest distance and add the hemispherical caps
                Length = (newDistance*pixsize + Width[0]/2 + Width[-1]/2)
                return Length,Width,Surface_area,Volume
    except:
        Length,Width,S,V = 0,0,0,0
    
        return Length,Width,Surface_area,V
    

def SingleCellLister(maskList:list) -> list:
    
    """From a list that contains instance segmentation images, obtain a list of individual masks

    Args:
        maskList(list): list of masks. If only one image is called, make sure to pass it as [image] for the function to run properly
    
    Returns:
        AllCells(list): list that contains a binary image of each single cell

    """

    AllCells = []

    for mask in maskList:
        reg = regionprops_table(mask,properties=['image'])
        Cells = [binary_fill_holes(image) for image in reg['image']]
        AllCells += Cells

    return AllCells

def SizeDataFrame(maskfilelist:list, from_files:bool=True, return_skeleton_paths:bool=False)->pd.DataFrame:

    """Function that analyses the images from a list of files and returns a pandas DataFrame with the cell size.
    In this function, cell IDs are not considered. If you're interested in that, please use the SizeDataFrame_Localizer function instead.
    It separates all the masks found in the images passed, then smoothes them by eroding and dilating them, adding a pad to prevent skeletons or distances to be on the edge
    Next, computes the skeleton and distance transform of each cell, which are then measured, returning a data table with all the metrics
    Further info (i.e., experiment, strain name, day) can be added later, but this is better done in a loop

    Args:
        maskfilelist(list): list of instance segmentation masks paths or already loaded images
        from_files(bool): If True, assumes that the masks are to be read first

    Returns:
        df(pd.DataFrame): pandas DataFrame with the cell size information of all the analyzed cells
    """
    if from_files:
        masks = [io.imread(maskfile) for maskfile in maskfilelist]

    else:
        masks = maskfilelist

    cells = SingleCellLister(masks)

    cells = [np.pad(binary_dilation(binary_erosion(cell)),pad_width=4) for cell in cells]
    skeletons = [skeletonize(cell) for cell in cells]
    distances = [distance_transform_edt(cell) for cell in cells]

    measures = [SkeletonMeasure(skel,dist,return_paths=return_skeleton_paths) for skel,dist in zip(skeletons,distances)]

    L = [metric[0] for metric in measures]
    w = [np.mean(metric[1]) for metric in measures]
    S = [metric[2] for metric in measures]
    V = [metric[3] for metric in measures]
    
    df = pd.DataFrame()

    df.insert(0,'Width',w)
    df.insert(1,'Length',L)
    df.insert(2,'SurfaceArea',S)
    df.insert(3,'Volume',V)

    return df

def SizeDataFrame_Localizer(maskfilelist:list)->pd.DataFrame:

    """Function that analyses the images from a list of files and returns a pandas DataFrame with the cell size.
    In this function, cell IDs as well as the name of the image are kept. Thus, the list passed must be a list of paths
    It separates all the masks found in the images passed, then smoothes them by eroding and dilating them, adding a pad to prevent skeletons or distances to be on the edge
    Next, computes the skeleton and distance transform of each cell, which are then measured, returning a data table with all the metrics
    Further info (i.e., experiment, strain name, day) can be added later, but this is better done in a loop
    For each image, an individual DatFrame is computed, and all of them are concatenated for the final output
    
    Args:
        maskfilelist(list): list of instance segmentation masks paths 

    Returns:
        final_df(pd.DataFrame): pandas DataFrame with the cell size information of all the analyzed cells, as well as the individual cell ID and the image they were obtained from
    """

    dfs = []

    masks = [io.imread(maskfile) for maskfile in maskfilelist]

    for i in range(len(masks)):

        mask = masks[i]
        
        reg = regionprops_table(mask,properties=['label','image'])
        cells = [binary_fill_holes(image) for image in reg['image']]
        
        cells = [np.pad(binary_dilation(binary_erosion(cell)),pad_width=4) for cell in cells]
        skeletons = [skeletonize(cell) for cell in cells]
        distances = [distance_transform_edt(cell) for cell in cells]
        
        measures = [SkeletonMeasure(skel,dist) for skel,dist in zip(skeletons,distances)]
        
        L = [metric[0] for metric in measures]
        w = [np.mean(metric[1]) for metric in measures]
        S = [metric[2] for metric in measures]
        V = [metric[3] for metric in measures]
        
        df = pd.DataFrame()

        mask_name = maskfilelist[i].split('/')[-1]

        df.insert(0,'ID',reg['label'])
        df.insert(0,'Mask',mask_name)
        df.insert(2,'Width',w)
        df.insert(3,'Length',L)
        df.insert(4,'SurfaceArea',S)
        df.insert(5,'Volume',V)

        dfs.append(df)

    if len(dfs) > 1:

        final_df = pd.concat(dfs,ignore_index=True)

        return final_df
    
    else:
        final_df = dfs[0]

        return final_df
    
