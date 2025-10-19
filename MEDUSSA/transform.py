import pandas as pd
import numpy as np
from scipy.stats import norm
from itertools import product
from pymc import HalfCauchy, Model, Normal, sample

def BayesianGLM(full_data:pd.DataFrame,metrics:list,GT_key:str,Target_key:str,sample_size:int=3000)->pd.DataFrame:

    """Run a Bayesian Generalized Linear Model (GLM). The GLM is computed for a normal linear relationship (y ~ mx + n + sigma).
    Keep in mind that this function was designed to transform values that exhibited a linear relationship between them. The function itself may not be suited for your specfic case, so be mindful.
    The slope (m) and intercept (n) are drawn from prior normal distributions (continuous positive and negative values)
    The error of the sampling (sigma) are drawn from a Half-Cauchy distribution (only positive continuous values)
    
    For input, it expects a dataset where each metric to transform has separate columns both for the ground-truth and target data, followed by an underscore and some string that helps differentiate between both. For example, on a comparison of GFP fluorescence, there would be a "GFP_input" column and a "GFP_prediction" column, where "GFP" is the metric, "input" ground-truth key and "prediction" the measurement key.

    Baseline, the function will sample 3000 parameter groups (m,n,sigma) four times, giving a total of 12000 possible value combinations. This can be changed with the "sample_size" argument.

    The output data frame contains five columns: Metric, Chain, Slope, Intercept, Sigma, and each row is a sampling event

    Args:
        - full_data(pd.DataFrame): a data frame containing the metrics of the ground-truth and target data
        - metrics(list): the metrics for which you want to compute the Bayesian GLM
        - GT_key(str): the column identifier for the ground-truth measurements
        - Target_key(str): the column identifier for the input measurements
        - sample_size(int): how many group of parameters are sampled in each chain (default 3000)
        
    Returns:
        - df_posteriors(pd.DataFrame): data frame with all the sampled [m,n,sigma] values for each metric

    """

    df_list = []

    for metric in metrics:

        y = np.array(full_data[f'{metric}_{GT_key}'])
        x = np.array(full_data[f'{metric}_{Target_key}'])

        RANDOM_SEED = 8927
        rng = np.random.default_rng(RANDOM_SEED)

        with Model() as model:

            ## Define priors
            sigma = HalfCauchy('sigma', beta=10)
            intercept = Normal('Intercept', 0, sigma=20)
            slope = Normal('slope', 0, sigma=20)

            ## Define likelihood
            likelihood = Normal('y', mu = intercept + slope * x, sigma=sigma, observed=y)

            ## For inference, draw N posterior samples
            idata = sample(sample_size)

        ## Obtain posterior distributions
        post_slope = np.array(idata.posterior.slope)
        post_intercept = np.array(idata.posterior.Intercept)
        post_sigma = np.array(idata.posterior.sigma)

        ## Each array has a shape of (4,3000), denoting the chains and draws, respectively
        ## Iterate through each chain to keep the information for sampling after

        for i in range(post_slope.shape[0]):
            
            mini_df = pd.DataFrame()
            mini_df.insert(0,'Sigma',post_sigma[i])
            mini_df.insert(0,'Intercept',post_intercept[i])
            mini_df.insert(0,'Slope',post_slope[i])
            mini_df.insert(0,'Chain',i)
            mini_df.insert(0,'Metric',metric)

            df_list.append(mini_df)

    df_posteriors = pd.concat(df_list,ignore_index=True)

    return df_posteriors

def ParamSampler(data_df:pd.DataFrame,posterior_df:pd.DataFrame,columns:list,metrics:list,nsamples:int=250)->pd.DataFrame:
    
    """Function to transform measurement data according to a linear relationship. Values of slope and intercept were obtained prior using the Bayesian_GLM function

    Args:
        data_df(pd.DataFrame): DataFrame that contains the data to be sampled
        posterior_df(list): DataFrame with the potential slope and intercept values obtained from Bayesian analysis. Must contain a 'Slope' and 'Intercept' columns
        columns(list): Columns in the DataFrame for subsetting
        metrics(list): List of the metrics that will be sampled (MUST BE NUMERIC)
        nsamples(int): How many times the pairs of slope and intercept will be randmoly sampled (default is 250)

    Returns:
        sampled_df(pd.DataFrame): DataFrame containing the sampled data, where each cell has for each metric a mean (mu) and standard deviation (sigma) value
    """

    column_levels = []
    
    ncols = len(columns)
    
    for c in columns:
    
        levels = np.unique(data_df[c])
    
        column_levels.append(levels)
    
    combinations = list(product(*column_levels))
    
    df_list = []
    
    for c in combinations:
    
        df = pd.DataFrame()
        
        positions = np.arange(0,ncols)
    
        subdf = data_df[data_df[columns[0]]==c[0]]
        
        if len(positions) > 1:
    
            for i in positions[1:]:
    
                subdf = subdf[subdf[columns[i]]==c[i]]
    
        i = 0
        for metric in metrics:
    
            x = np.array(subdf[metric])
            
            subdf_posterior = posterior_df[posterior_df['Metric'] == metric]
            m = np.array(subdf_posterior['Slope'])
            n = np.array(subdf_posterior['Intercept'])
    
            sample_size = nsamples
        
            idxs = [np.random.randint(low=0,high=12000,size=sample_size) for i in x]
    
            m_ = [m[idx] for idx in idxs]
            n_ = [n[idx] for idx in idxs]
    
            y = [np.array(m__)*x__ + np.array(n__) for m__,x__,n__ in zip(m_,x,n_)]
    
            y_mu = [y_.mean() for y_ in y]
            y_sigma = [y_.std() for y_ in y]
    
            i
            df.insert(i,f'{metric}_mu',y_mu)
            i+=1
            df.insert(i,f'{metric}_sigma',y_sigma)
            i+=1
    
        df.insert(0,f'{columns[0]}',c[0])
    
        if len(positions) > 1:
    
            for i,j in enumerate(positions[1:]):
    
                df.insert(i+1,f'{columns[j]}',c[j])
        
        df_list.append(df)
    
    sampled_df = pd.concat(df_list)

    return sampled_df

def ConfIntSampler(df:pd.DataFrame,columns:list,metric:str,interval:float=0.99)->pd.DataFrame:

    """Function to sample measurement data according to a defined interval. It can subset the data in multiple combinations

    Args:
        df(pd.DataFrame): DataFrame that contains the data to be sampled
        columns(list): Columns in the DataFrame for subsetting
        metric(str): Name of the metric to be sampled
        interval(float): the interval that will be sampled (i.e., if 0.95, the 2.5% upper and 2.5% lower points are discarded). Must be between 0 and 1

    Returns:
        sampled_df(pd.DataFrame): DataFrame containing the sampled data with all the conditions used for subsetting
    """

    m_samples = []
    
    column_levels = []

    ncols = len(columns)

    for c in columns:

        levels = np.unique(df[c])
    
        column_levels.append(levels)

    combinations = list(product(*column_levels))

    for c in combinations:

        idxs = np.arange(0,ncols)
    
        subdf = df[df[columns[0]]==c[0]]
    
        if len(idxs) > 1:
    
            for i in idxs[1:]:
    
                subdf = subdf[subdf[columns[i]]==c[i]]
    
        arr = np.array(subdf[metric])
        ci = norm(*norm.fit(arr)).interval(interval) 
    
        subdf_m = subdf[subdf[metric]>=ci[0]]
        subdf_m = subdf_m[subdf_m[metric]<=ci[1]]
    
        m_samples.append(subdf_m)

    sampled_df = pd.concat(m_samples)

    return sampled_df