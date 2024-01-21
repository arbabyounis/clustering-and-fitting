# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 23:31:47 2024

@author: arbab
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler
import sklearn.cluster as cluster
import scipy.optimize as opt
import matplotlib.gridspec as gridspec 
import itertools

# Set up a consistent style for plotting
plt.style.use('seaborn')
# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def world_bank_reader(wb_file_name):
    """Reads data from the World Bank CSV file."""
    climate_change_df = pd.read_csv(wb_file_name, skiprows=4) 
    climate_change_df.drop(climate_change_df.columns[[-1]], axis=1, inplace=True)
    
    climate_year_df = climate_change_df[:]  
    climate_country_df = climate_change_df.groupby(['Country Name']).aggregate('first').transpose()
    
    return climate_country_df, climate_year_df

# PART 1 - Normalization & Clustering Functions

def norm_fun(data):
    """Returns array normalized to [0,1]."""
    minimum_val = np.min(data)
    maximum_val = np.max(data)
    norm_data = (data - minimum_val) / (maximum_val - minimum_val)
    return norm_data

def normalize_df(norm_data):
    """Normalizes all columns of the DataFrame to [0,1]."""
    for cc_column in norm_data.columns[1:]:  # excluding the first column
        norm_data[cc_column] = norm_fun(norm_data[cc_column])
    return norm_data

def square_plot(data, xaxis, yaxis):
    """Produces a square plot of two columns of a DataFrame."""
    plt.plot(data[xaxis], data[yaxis], "o", markersize=6)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)

def k_mean(data, xaxis, yaxis, n, title):
    """Applies k-means clustering and plots the result."""
    kmeans = cluster.KMeans(n_clusters=n)
    data_fit = data[[xaxis, yaxis]].copy()
    kmeans.fit(data_fit)     

    cluster_labels = kmeans.labels_
    cluster_center = kmeans.cluster_centers_

    col = ["blue", "red", "green", "magenta", "yellow", "orange"]
    
    for l in range(n):     
        plt.plot(data_fit[xaxis][cluster_labels==l], data_fit[yaxis][cluster_labels==l],
                 "o", markersize=8, color=col[l])
    
    for iter_cent in range(n):
        xcluster, ycluster = cluster_center[iter_cent,:]
        plt.plot(xcluster, ycluster, "dk", markersize=12, color='white')
    
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)

# Define Global Variables & Lists for Data Manipulation
forest_area = "AG.LND.FRST.K2"
co2 = "EN.ATM.CO2E.PC"
population = "SP.POP.TOTL"
energy_use = "EG.USE.PCAP.KG.OE"
arable_land = "AG.LND.ARBL.ZS"
renewable_euse = "EG.FEC.RNEW.ZS"

# Define mapping for Indicator codes
indicator_replace = {
    "EN.ATM.CO2E.PC": "CO2",
    "AG.LND.FRST.K2": "Forest_Area",
    "SP.POP.TOTL": "Population",
    "EG.USE.PCAP.KG.OE": "Energy_use",
    "AG.LND.ARBL.ZS": "Arable_Land",
    "EG.FEC.RNEW.ZS": "Renewable_Fuel"
}

# Define Indicator map for DataFrame
indicator_map = ["EN.ATM.CO2E.PC", "AG.LND.FRST.K2", "SP.POP.TOTL", 
                  "EG.USE.PCAP.KG.OE", "AG.LND.ARBL.ZS", "EG.FEC.RNEW.ZS"]

# Assign World Bank Data file name to a variable
wb_file_name = 'Data_set.csv'

# Use World Bank Reader function to generate 2x data frame as per the assignment requirement
country_df, year_df = world_bank_reader(wb_file_name)

# Filter out the required indicators for the analysis from the Indicator Map list
year_df_new = year_df[year_df['Indicator Code'].isin(indicator_map)]

# Replace Not Identifiable Indicator codes with known Identifiers
year_df_new["Indicator Code"].replace(indicator_replace, inplace=True)

year_df_new = year_df_new.copy(deep=True)

# Drop Country Name & Indicator Name as they are already defined
year_df_new = year_df_new.drop(['Country Code', 'Indicator Name'], axis=1, inplace=False)
country_data = year_df_new.reset_index()

# Normalize the index for easy analysis
country_data.drop(['index'], axis=1, inplace=True)

stat_data = country_data.groupby(['Country Name', 'Indicator Code']).aggregate('mean')
df_env_f = stat_data.stack().unstack(level=1)

stat_data2 = df_env_f.groupby(['Country Name']).aggregate('mean')
stat_data_orig = stat_data2.reset_index()

# Handle null values
stat_data_orig['CO2'].fillna(stat_data_orig['CO2'].mean(), inplace=True)
stat_data_orig['Forest_Area'].fillna(stat_data_orig['Forest_Area'].mean(), inplace=True)
stat_data_orig['Arable_Land'].fillna(stat_data_orig['Arable_Land'].mean(), inplace=True)
stat_data_orig['Energy_use'].fillna(stat_data_orig['Energy_use'].mean(), inplace=True)
stat_data_orig['Renewable_Fuel'].fillna(stat_data_orig['Renewable_Fuel'].mean(), inplace=True)

# Set up gridspec figure
fig = plt.figure(figsize=(15, 8), constrained_layout=True)

# Setting the Columns & Rows to the Grid Spec object
gs = fig.add_gridspec(nrows=2, ncols=3) 

ax1 = fig.add_subplot(gs[0, 0])
k_mean(stat_data_orig, "CO2", "Population", 4, "Plot 1 Cluster Membership")

ax2 = fig.add_subplot(gs[0, 1])
k_mean(stat_data_orig, "Forest_Area", "Population", 4, "Plot 2 Cluster Membership")

ax3 = fig.add_subplot(gs[1, 0])
k_mean(stat_data_orig, "Renewable_Fuel", "Population", 4, "Plot 3 Cluster Membership")

# PART 2 - Fitting Functions

def exp_growth(time, scale_val, growth_val):
    """Calculates exponential function."""
    xf = scale_val * np.exp(growth_val * (time - 1960)) 
    return xf

def err_ranges(data, modl, parameter, sigma_val):
    """Calculates upper and lower limits for the function."""
    lower_lim = modl(data, *parameter)
    upper_lim = lower_lim
    
    uplow_lim = []
    
    for q, t in zip(parameter, sigma_val):
        pmin_val = q - t
        pmax_val = q + t
        uplow_lim.append((pmin_val, pmax_val))
        
    pmix_val = list(itertools.product(*uplow_lim))
    
    for q in pmix_val:
        y_modl = modl(data, *q)
        lower_lim = np.minimum(lower_lim, y_modl)
        upper_lim = np.maximum(upper_lim, y_modl)
        
    return lower_lim, upper_lim

def fit_plot(data, xaxis, yaxis, fit_param, xlbl, ylbl, title, cl1, cl2):
    """Plots the fitting plot."""
    plt.figure()
    plt.plot(data[xaxis], data[yaxis], label="Data", color=cl1)
    plt.plot(data[xaxis], data[fit_param], label="Fit", color=cl2)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.show()

def fit_err_plot(data, low, up, xaxis, yaxis, fit_param, xlbl, ylbl, title, cl1, cl2, cl3):
    """Plots the fitting plot with error/confidence values."""
    plt.figure()
    plt.title(title)
    plt.plot(data[xaxis], data[yaxis], label="Data", color=cl1)
    plt.plot(data[xaxis], data[fit_param], label="Fit", color=cl2)
    plt.fill_between(data[xaxis], low, up, alpha=0.9, color=cl3)
    plt.legend()
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.show()

def best_fit_plot(data, env_data, forcast, low, up, xaxis, yaxis, xlbl, ylbl, title, col1, col2, col3):
    """Plots the best fitting plot with error/confidence values."""
    plt.figure()
    plt.title(title)
    plt.plot(data[xaxis], data[yaxis], label=ylbl, color=col1)
    plt.plot(env_data, forcast, label="Forecast", color=col2)
    plt.fill_between(env_data, low, up, alpha=0.9, color=col3)
    plt.legend()
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.show()

# Extracting population data of India & China from the World Bank DataFrame
pop_stat = df_env_f.loc[['India'], ['Population']]
pop_stat_chn = df_env_f.loc[['China'], ['Population']]

# Removing the indexing for the time series analysis
pop_stat_orig = pop_stat.reset_index()
pop_stat_orig_chn = pop_stat_chn.reset_index()

# Renaming the default column name to 'Years'
pop_stat_orig.rename({'level_1':'Years'}, axis=1, inplace=True)
pop_stat_orig_chn.rename({'level_1':'Years'}, axis=1, inplace=True)

pop_stat_orig["Years"] = pd.to_numeric(pop_stat_orig["Years"])

# Estimated turning year: 1985 to 2000
# The growth observed in 2000 is when the population reaches 1.05 Billion
# Population in 1985 was about 780 million
# Best fitting parameters for the growth model
popt = [2e9, 0.05, 1985]

# Apply the logistics model function with indicators
pop_stat_orig["pop_log"] = logistics(pop_stat_orig["Years"], *popt)

# Fit exponential growth with default parameters
popt, covar = opt.curve_fit(exp_growth, pop_stat_orig["Years"], pop_stat_orig["Population"])

# Parameters for exponential growth model
popt = [4e8, 0.02]

# Apply the exponential growth model with the obtained parameters
pop_stat_orig["pop_exp"] = exp_growth(pop_stat_orig["Years"], *popt)

# Fit exponential growth by applying the best Fit parameters obtained above
# We use curve_fit for fitting the exponential model
popt, covar = opt.curve_fit(exp_growth, pop_stat_orig["Years"], 
                            pop_stat_orig["Population"], p0=[4e8, 0.02])

pop_stat_orig["pop_exp"] = exp_growth(pop_stat_orig["Years"], *popt)

# Extract sigma values from the diagonal of the Covariance Matrix
sigma = np.sqrt(np.diag(covar))

# Call the pre-defined Error Range Function to find the error ranges
low, up = err_ranges(pop_stat_orig["Years"], exp_growth, popt, sigma)

# Call pre-defined Plotting function for the Fitted Model
fit_err_plot(pop_stat_orig, low, up, "Years", "Population", "pop_log", "Years", "Population", "Logistics Model with Error Ranges", 'y', 'r', 'b')

print("Sigma", sigma)


