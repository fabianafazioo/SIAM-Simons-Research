# -*- coding: utf-8 -*-

import pdb
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from scipy import stats
from itertools import combinations
from sklearn.datasets import load_iris
from scipy.stats import anderson
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from scipy.stats import kruskal

"""
The function FakeMeasures does the following :
    
  Parameters:
    - features: a Dataframe of all the columns except the label column 
    - label: an Dataframe column that represents the label column 
    - optional: a boolean list where TRUE indicates a CATEGORICAL variable
        This argument specifies which columns should be considered categorical.
        If optional is not provided, then the program will ONLY consider floats as
        NUMERICAL and ALL other data types (integers, objects/strings) as categorical. 
    
  What does this function do?:
    - This function takes 1 dataset. It tests the dataset through 3 different tests 
    - Correlation between columns (Feature vs Feature) is for NUMERICAL features & finds their correlation
    - Chi-square test (Feature vs Feature) is for CATEGORICAL features and compares only the features columns
    - Chi-square test (Feature vs Label) is for CATEGORICAL features and compares the features columns to the label column
    
  Output:
    - Correlation between columns (Feature vs Feature) returns a correlation matrix for NUMERICAL features
    - Chi-square test (Feature vs Feature) returns 2 Dataframes for every combination of CATEGORICAL features: 
        a matrix of the Chi-square statistics and a matrix of the p-values
    - Chi-square test (Feature vs Label) returns a Dataframe of the Chi-square statistics and p-values in 2 columns

"""

# STEPS TO DO IN A     SEPARATE     FILE:
    
################## Import function ############################################
# import sys
# sys.path.append('C:/Users/...YOUR Fake_Measures.py FILE PATH HERE.../')
# import Fake_Measures

################## Import data ################################################
# df = pd.read_csv("C:/Users/...YOUR data FILE PATH HERE")

################## Data cleaning ##############################################
# Whatever changes you make to your data set

################## Split dataset into X_train and y_train #####################
# X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, YOUR COLUMN INDEXES], df.iloc[:, YOUR LABEL INDEX], test_size=0.2, random_state=42)

################## Running Measure_Patterns() #################################
# Fake_Measures.FakeMeasures(X_train, y_train)

#################### DO NOT UNCOMMENT!!!!!!!! #################################

# Function Measure_Patterns begins here!
def FakeMeasures(features, label, optional=None):
    
    # Initialize empty dataframes for NUMERICAL and CATEGORICAL data
    numerical_df = pd.DataFrame()
    categorical_df = pd.DataFrame()
    
    # Check if the data type is provided for columns
    if optional is None:
        print("Optional parameter not provided. Assuming integers values are categorical")
    
        # Splitting features into NUMERICAL subset 
        print("\nNumerical DataFrame:")
        numerical_df = features.select_dtypes(include = ['float', 'float64'])
        print(numerical_df)

        # Splitting features into CATEGORICAL subset 
        print("\nCategorical DataFrame:")
        categorical_df = features.select_dtypes(exclude=['float', 'float64'])
        print(categorical_df)
    else:
        # Create empty NUMERICAL & CATEGORICAL data frames
        numerical = []
        numerical_colnames = []
        categorical = []
        categorical_colnames = []
        
        # Check that length of optional = # of columns in features
        # Optional is the column type for features, so the lengths should be equal
        if len(optional) == len(features.columns):
            # For all the values in optional
            for i in range(len(optional)):
                if optional[i] == True:
                    # Save CATEGORICAL column in CATEGORICAL list
                    categorical.append(features.iloc[:,i])
                    # Save CATEGORICAL column name
                    categorical_colnames.append(features.columns[i])
                else: 
                    # Save NUMERICAL column in NUMERICAL list
                    numerical.append(features.iloc[:,i])
                    # Save NUMERICAL column name
                    numerical_colnames.append(features.columns[i]) 
            # Turn transposed arrays into dataframes
            numerical_df = pd.DataFrame(np.transpose(numerical))
            categorical_df = pd.DataFrame(np.transpose(categorical))
            
            # Re-attach the column names to numerical_df & categorical_df 
            if numerical_colnames:  # Check if there are numerical columns
                numerical_df.columns = numerical_colnames # Reassign names
            else:
                numerical_df = pd.DataFrame(columns=[])  # Empty dataframe with no columns
            
            if categorical_colnames:  # Check if there are categorical columns
                categorical_df.columns = categorical_colnames # Reassign names
            else:
                categorical_df = pd.DataFrame(columns=[])  # Empty dataframe with no columns

            print("Numerical DF:")
            print(numerical_df)
            print("Categorical Df")
            print(categorical_df)
            
        else:
            print("The length of features and optional are different.")


################# Correlation between columns (NUMERICAL) Code ################
    # Takes the features data to find correlation between NUMERICAL features
    def num_corr(features_numerical):
        matrix = features_numerical.corr(method='pearson')
        print("---------------------------Correlation Matrix------------------------- \n", matrix)
        return matrix
     
    #Calls the function so the matrix prints out    
    num_corr(numerical_df)
    
##################### Chi-Square (F vs F) Code ################################
    
    print("\n------------------Chi-Squared for Features v. Features-----------------------")
    # Finds dependency between all CATEGORICAL features
    def chi_squared_fvf(features):
            
        # Extract variable names
        variable_names = list(features.columns)
    
        # Initialize matrices to store chi-squared and p-values
        num_variables = len(variable_names)
        chi_squared = np.zeros((num_variables, num_variables))
        p_values = np.zeros((num_variables, num_variables))
    
        # Creates an empty boolean array for contingency table cells <5
        below_5 = []
    
        # Compute chi-squared and p-values for each pair of variables
        for i, j in combinations(range(num_variables), 2):
            
            # Creates contigency table of feature i v. feature j
            contingency_table = pd.crosstab(features.iloc[:, i], features.iloc[:, j])
            
            # Check if any cell in the contingency table is below 5 & appends
            below_5.append((contingency_table < 5).any().any())
            
            # Compute chi-squared and p-values
            chi2 = stats.chi2_contingency(contingency_table)[0]
            p = stats.chi2_contingency(contingency_table)[1]
                
            # Assign results to chi_squared and p_values matrices
            chi_squared[i, j] = chi2
            chi_squared[j, i] = chi2  # Assign to symmetric position in the matrix
            p_values[i, j] = p
            p_values[j, i] = p  # Assign to symmetric position in the matrix
    
        # Create a DataFrame with variable names as index and columns
        chi_squared_df = pd.DataFrame(chi_squared, index=variable_names, columns=variable_names)
        p_values_df = pd.DataFrame(p_values, index=variable_names, columns=variable_names)
    
        # Print warning if any cells are below 5
        if any(below_5):
            print("WARNING: The validity of this chi-squared test may be violated as there are \n         cells below 5 in at least one contingency table of observed values.") 
        
        # Printing the matrix-like output with variable names
        print("Chi-Squared Statistics:")
        print(chi_squared_df)
        print("\nP-Values:")
        print(p_values_df)
        return chi_squared_df, p_values_df
        
    chi_squared_fvf(categorical_df)
    
##################### Chi-Square (F vs label column) Code #####################
    
    print("\n------------------------Chi-Square (F vs label column)------------------------")
    # Finds dependency between all CATEGORICAL features & the label
    def chi_squared_fvl(features, label):
            
        # Combining features and label
        df = features.copy()
        df[label.name] = label  # Preserving the original column name of label (debugged)
        df = df[[col for col in df.columns if col != label.name] + [label.name]] # Label becomes last column
    
        # Number of features, excluding label
        var_count = len(df.columns)-1
    
        # Creates an empty array for Chi2 and P-values
        results = []
    
        # Creates an empty boolean array for contingency table cells <5
        below_5 = []
    
        for i in range(0, var_count):
    
            # Create contigency table of all features v. label
            contingency_table = pd.crosstab(df.iloc[:, i], df.iloc[:,-1])
            
            # Check if any cell in the contingency table is below 5 & appends
            below_5.append((contingency_table < 5).any().any())
                
            # Compute chi-squared and p-values
            chi2 = stats.chi2_contingency(contingency_table)[0]
            p = stats.chi2_contingency(contingency_table)[1]
                
            # Append results to the list
            results.append({
                "Feature": df.columns[i],
                "Chi Squared Statistic": chi2,
                "P-Value": p})
    
        # Create a dataFrame from the results
        results_df = pd.DataFrame(results)
    
        # Print warning if any cells are below 5
        if any(below_5):
            print("WARNING: The validity of this chi-squared test may be violated as there are \n         cells below 5 in your contingency table of observed values.") 
        
        # Print the dataFrame
        print("Label:", df.columns.values[-1])
        print(results_df.to_string(index=False))
        return results_df
        
    chi_squared_fvl(categorical_df, label)
    
############################# ANOVA (Feature vs Feature) ######################
    print("\n------------------ANOVA for Features v. Features-----------------------")
    # Finds dependency between all features in features
    def anova_fvf(features):
            
        # Extract variable names
        variable_names = list(features.columns)
    
        # Initialize matrices to store chi-squared and p-values
        num_variables = len(variable_names)
        f_stats = np.zeros((num_variables, num_variables))
        p_values = np.zeros((num_variables, num_variables))
    
        # Compute chi-squared and p-values for each pair of variables
        for i, j in combinations(range(num_variables), 2):
    
            # Compute ANOVA: f-statistics and p-values
            f, p = f_oneway(features.iloc[:, i], features.iloc[:, j])
                
            # Assign results to f_stats and p_values matrices
            f_stats[i, j] = f
            f_stats[j, i] = f  # Assign to symmetric position in the matrix
            p_values[i, j] = p
            p_values[j, i] = p  # Assign to symmetric position in the matrix
    
        # Create a DataFrame with variable names as index and columns
        f_stats_df = pd.DataFrame(f_stats, index=variable_names, columns=variable_names)
        p_values_df = pd.DataFrame(p_values, index=variable_names, columns=variable_names)
    
        # Printing the matrix-like output with variable names
        print("\nF-Statistics:")
        print(f_stats_df)
        print("\nP-Values:")
        print(p_values_df)
        
    anova_fvf(features)
    
############################# ANOVA (Feature vs Label) ######################
    print("\n------------------ ANOVA (Feature vs Label) -----------------------")
    
    # Finds dependency between all features in features & the label  
    def anova_fvl(features, label):
        
        # Combining features and label
        df = features.copy()
        df[label.name] = label  # Preserving the original column name of label (debugged)
        df = df[[col for col in df.columns if col != label.name] + [label.name]] # Label becomes last column
    
        # Number of features, excluding label
        var_count = len(features.columns)-1
    
        # Creates an empty array for f-statistic and P-values
        results = []
    
        for i in range(0, var_count):
            
            # Compute ANOVA
            f_statistic, p_value = f_oneway(df.iloc[:,i], df.iloc[:,-1])
            
            # Save p-value significance into list
            if p_value < 0.05:
                significance = "Significant"
            else:
                significance = "Not Significant"
               
            # Append results to the list
            results.append({
                "Feature": df.columns[i],
                "F-Statistic": f_statistic,
                "P-Value": p_value, 
                "Significance": significance})
    
        # Create a dataFrame from the results
        results_df = pd.DataFrame(results)
        
        # Print the dataFrame
        print("Label:", df.columns.values[-1])
        print(results_df.to_string(index=False))
    
    # Run ANOVA
    anova_fvl(features, label)
    
################## KRUSKAL-WALLIS H Test (FvF) #######################################
    
    print("\n------------------ Kruskal-Wallis H Test (Feature vs Feature) -----------------------")
    # Determines if all features in features have same mean via ranks
    def KWH_fvf(features):
        # Extract variable names
        variable_names = list(features.columns)
    
        # Initialize matrices to store KWH-statistic and p-values
        num_variables = len(variable_names)
        kwh_stats = np.zeros((num_variables, num_variables))
        p_values = np.zeros((num_variables, num_variables))
    
        # Compute KWH-statistic and p-value for each pair of variables
        for i, j in combinations(range(num_variables), 2):
            try:
                # Checks if both columns are identical; KWH cannot run if so
                if np.array_equal(features.iloc[:, i], features.iloc[:, j]):
                    kwh_stats[i, j] = np.nan # Replacing that matrix value with NaN instead of running KWH test
                    kwh_stats[j, i] = np.nan
                    p_values[i, j] = np.nan
                    p_values[j, i] = np.nan
                else:
                    # Compute KRUSKA-WALLIS H Test
                    kwh, p = kruskal(features.iloc[:, i], features.iloc[:, j])
                    
                    # Assign results to kwh_stats and p_values matrices
                    kwh_stats[i, j] = kwh
                    kwh_stats[j, i] = kwh  # Assign to symmetric position in the matrix
                    p_values[i, j] = p
                    p_values[j, i] = p  # Assign to symmetric position in the matrix
            except ValueError as e:
                print(f"Error: {e}") # if TRY fails, print error 
    
        # Create a DataFrame with variable names as index and columns
        kwh_stats_df = pd.DataFrame(kwh_stats, index=variable_names, columns=variable_names)
        p_values_df = pd.DataFrame(p_values, index=variable_names, columns=variable_names)
    
        # Printing the matrix-like output with variable names
        print("\nKruskal-Wallis H statistic:")
        print(kwh_stats_df)
        print("\nP-Values:")
        print(p_values_df)
        
    KWH_fvf(features)
    
################## KRUSKAL-WALLIS H Test #######################################

    print("\n------------------ Kruskal-Wallis H Test (Feature vs Label) -----------------------")
    
    # Determines if all features in features & the label have same mean via ranks
    def KWH_fvl(features, label):
        
        # Combining features and label
        df = features.copy()
        df[label.name] = label  # Preserving the original column name of label (debugged)
        df = df[[col for col in df.columns if col != label.name] + [label.name]] # Label becomes last column
    
        # Number of features, excluding label
        var_count = len(features.columns)-1
        
        # Creates an empty array for f-statistic and P-values
        results = []
    
        for i in range(0, var_count):
            
            # Compute KRUSKA-WALLIS H Test
            kwh_statistic, p_value = kruskal(df.iloc[:,i], df.iloc[:,-1])
            
            # Save p-value significance into list
            if p_value < 0.05:
                significance = "Significant"
            else:
                significance = "Not Significant"
               
            # Append results to the list
            results.append({
                "Feature": df.columns[i],
                "Kruskal-Wallis H statistic": kwh_statistic,
                "P-Value": p_value, 
                "Significance": significance})
    
        # Create a dataFrame from the results
        results_df = pd.DataFrame(results)
        
        # Print the dataFrame
        print("Label:", df.columns.values[-1])
        print(results_df.to_string(index=False))
    
    # Run ANOVA
    KWH_fvl(features, label)

######################### Histogram/Graphing #################################
    print("------------------------Histogram/Graphing-----------------------------") 
    
    # Ensure data is 2D
    if numerical_df.ndim == 1:
        numerical_df = numerical_df.reshape(-1, 1)  # Reshape 1D array to 2D array with one column
    
    # Number of features (columns) in the dataset
    numerical_num_features = numerical_df.shape[1]
    
    # Loop through each numerical feature
    for feature_idx in range(numerical_num_features):
        # Extract the current feature data (column)
        feature_df = numerical_df.iloc[:, feature_idx]
    
        # Compute histogram with 10 bins
        hist, bin_edges = np.histogram(feature_df, bins=10)
    
        # Print feature number
        print(f"Feature {feature_idx + 1}:")
        
        # Print bin edges
        print("Bin Edges:", bin_edges)
    
        # Store bin heights in a list
        bin_heights = []
        bin_heights.extend(hist)
        print("Array with bin heights:", bin_heights)
    
        # Store bin probabilities in a list and normalize
        bin_probs = []
        bin_probs.extend(hist)
        bin_probs = np.array(bin_probs) / sum(bin_heights)
        print("Array with bin probabilities:", bin_probs)
    
        # Loop through each bin to print range and probabilities
        for i in range(len(hist)):
            bin_range = f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}"  # Bin range
            bin_probability = hist[i] / sum(hist)  # Bin probability
            print(f"Bin {i + 1} ({bin_range}): Height = {hist[i]}, Probability = {bin_probability:.2f}")
    
        # Separator between features for clarity
        print("\n" + "="*50 + "\n")
    
    # Calculate and store probabilities for each categorical column
    print("Proportions for Label for Categorical Columns:")
    
    for column in categorical_df.columns:
        value_counts = categorical_df[column].value_counts(normalize=True).sort_index()
        # print(f"Probabilities for Categorical Column {column}:")
        print(value_counts)
        print()  # Add an empty line for separation    
        


