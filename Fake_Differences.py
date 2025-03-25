#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:55:06 2024

The function FakeDifferences does the following :
    
  Parameters:
    - data1 is the sythetic dataset (before augmentation)
    - data2 is the augmented dataset
    - optional is a boolean list where TRUE indicates a CATEGORICAL variable.
    This will check if the columns selected are categorical (integers and strings) 
    or numerical (floats). If optional is not provided, then the program will assume that 
    the column has integers values, therefore it will be considered categorical
    
    
  What does this function do?:
    - This function takes 2 datasets and compares the differences between them
    - Mann-Whitney U Test is for (Feature  Distribution) and compares the historgram 
    distributions of each numerical column before vs after augmentation
    - Categorical proportions test is for (Feature Distribution) of categorical variables
    and compares the subcategory proportions before vs. after augmentation by finding
    the Euclidean norm of the difference between the arrays of subcategory proportions
    - Correlation between columns test is for (Feature vs Feature) and tests the 
    Frobenius norm which computes the difference between the correlation matrices
    - Chi-square test is for (Feature vs Feature) which compares only the feature columns in the dataset 
    that are categorical (excluding the label column). 
    - Chi-square test is for (Feature vs Label) which compares the categorical columns to 
    the label column
    
  Output:
    - The Mann-Whitney U Test outputs a visual representation of the historgram for every
    numerical column, where the blue color represents the overlap between the historgram
    of before vs after augmentation. It also outputs a U-Statistic and a p-value, which indicates
    whether the difference in the distributions is significant or not
    - The categorical proportions test prints out the proportions of observations in each of the
    subcategories for every categorical column before and after augmentation and it also
    prints the Euclidean norm of the difference between the two
    - Correlation between columns test outputs 2 matrices correlations that only takes
    the numerical columns in the dataset. This outputs the differences between both correlations
    by printing the absolute error and relative error betweem them
    - Chi-square test (Feature vs Feature) outputs the Chi-square significance values in a dataframe, 
    and the p-value in another dataframe. Both take only the catgorical columns in the dataset. These
    dataframes change to a True and False dataframe which only uses the p-values. 
    The final output is the count of changes in the True and False table.
       - If the p-value is less than 0.05 there is no relationship (This is the H0, prints TRUE)
       - If the p-value is greater than 0.05 there us a relationship (This is the Ha, prints FALSE)
    - Chi-square test (Feature vs Label) outputs the the Chi-square significance values and p-values
    in 2 columns. Both take only the catgorical columns in the dataset. The test only grabs the p-values
    and changes them to True and False. 
    The final output is the count of changes in the True and False table.
       - If the p-value is less than 0.05 there is no relationship (This is the H0, prints TRUE)
       - If the p-value is greater than 0.05 there us a relationship (This is the Ha, prints FALSE)
    
"""

# Upload libraries
import numpy as np
import pandas as pd
from numpy import linalg
from itertools import combinations
from scipy import stats
import matplotlib.pyplot as plt

# STEPS TO DO IN A     SEPARATE     FILE:
    
################## Import function ############################################
# import sys
# sys.path.append('/Users/...YOUR FILE PATH HERE FOR FAKE_DIFFERENCES.../')
# import Fake_Differences

################## Import data ################################################
# df = pd.read_csv("C:/Users/...YOUR dataset FILE PATH HERE")

################## Data cleaning ##############################################
# Whatever changes you make to your data set

################# Categorical and Numerical Columns ###########################
# Whatever subsetting you make to your columns

################## Running FakeDifferences() #################################
# Fake_Differences.FakeDifferences(data1, data2)

#################### DO NOT UNCOMMENT!!!!!!!! #################################


def FakeDifferences(data1, data2, optional=None):
    # Function to split data into numerical and categorical dataframes
    def split_data(data, optional):
        numerical_df = pd.DataFrame()
        categorical_df = pd.DataFrame()
        
        if optional is None:
            print("Optional parameter not provided. Assuming integer values are categorical.")
            numerical_df = data.select_dtypes(include=['float', 'float64'])
            categorical_df = data.select_dtypes(exclude=['float', 'float64'])
        else:
            numerical = []
            numerical_colnames = []
            categorical = []
            categorical_colnames = []

            if len(optional) == len(data.columns):
                for i in range(len(optional)):
                    if optional[i] == False:  # FALSE = Numerical
                        numerical.append(data.iloc[:, i])
                        numerical_colnames.append(data.columns[i])
                    else:
                        categorical.append(data.iloc[:, i])  # TRUE = Categorical
                        categorical_colnames.append(data.columns[i])

                # Convert lists to DataFrames only if they contain data
                if numerical:
                    numerical_df = pd.DataFrame(np.array(numerical).T, columns=numerical_colnames)
                if categorical:
                    categorical_df = pd.DataFrame(np.array(categorical).T, columns=categorical_colnames)
            else:
                print("Error: The length of 'optional' does not match the number of columns in the dataset.")

        return numerical_df, categorical_df

    # Split both datasets
    numerical_df1, categorical_df1 = split_data(data1, optional)
    numerical_df2, categorical_df2 = split_data(data2, optional)

    # Check if any dataframe is empty
    if numerical_df1.empty:
        print("Warning: Numerical DataFrame 1 is empty.")
    if categorical_df1.empty:
        print("Warning: Categorical DataFrame 1 is empty.")
    if numerical_df2.empty:
        print("Warning: Numerical DataFrame 2 is empty.")
    if categorical_df2.empty:
        print("Warning: Categorical DataFrame 2 is empty.")

    # Check if numerical and categorical dataframes match across datasets
    if set(numerical_df1.columns) != set(numerical_df2.columns):
        print("Warning: Numerical DataFrames have different column structures.")
    if set(categorical_df1.columns) != set(categorical_df2.columns):
        print("Warning: Categorical DataFrames have different column structures.")

    # Print numerical and categorical dataframes for debugging
    print("\nNumerical DataFrame 1:")
    print(numerical_df1)
    print("\nCategorical DataFrame 1:")
    print(categorical_df1)
    
    print("\nNumerical DataFrame 2:")
    print(numerical_df2)
    print("\nCategorical DataFrame 2:")
    print(categorical_df2)
    
############################## Feature Distributions  ##################################

    def feat_dist(numerical_df1, categorical_df1, numerical_df2, categorical_df2):
        
        # Histogram function
        def generate_distribution_histogram(dataframe, column_name, number_bins, 
                                    alpha, color, edgecolor, label_name,
                                    density = True,
                                    title='', x_axis_label='', y_axis_label=''):
    
            plt.hist(dataframe[column_name], bins = number_bins, alpha = alpha, color = color, 
                  edgecolor = edgecolor, label = label_name, density = density)
            plt.title(title)
            plt.xlabel(x_axis_label)
            plt.ylabel(y_axis_label)
            plt.legend(loc='upper left')
        # Mann-Whitney U Test function
        def mann_whitney_u_test(distribution_1, distribution_2):
            try:
                u_statistic, p_value = stats.mannwhitneyu(distribution_1, distribution_2)
            
                #Print the results
                print('U-Statistic: ', u_statistic)
                print('p-value: ', p_value)
                print()
            except ValueError as e:
                print("Mann-Whitney U Test could not be performed:", e)

        # Check for empty dataframes across numerical datasets
        if numerical_df1.empty or numerical_df2.empty:
            print("Skipping numerical distribution comparison due to missing numerical data.\n")
        else:
            num_X_train1 = numerical_df1.iloc[:, :-1] if numerical_df1.shape[1] > 1 else numerical_df1
            num_X_train2 = numerical_df2.iloc[:, :-1] if numerical_df2.shape[1] > 1 else numerical_df2
            
            # Running feature distribution analyses for NUMERICAL data
            for column in num_X_train1.columns.intersection(num_X_train2.columns):
                
                # Calculate common bins based on the range of values in both datasets for this column
                combined_min = min(np.min(num_X_train1[column]), np.min(num_X_train2[column]))
                combined_max = max(np.max(num_X_train1[column]), np.max(num_X_train2[column]))
                bins = np.linspace(combined_min, combined_max, 10)  # Adjust the number of bins as needed
        
                generate_distribution_histogram(dataframe = num_X_train1, 
                                                column_name = column, 
                                                number_bins = bins, 
                                                alpha = 0.8, 
                                                color = 'teal', 
                                                edgecolor = 'black',
                                                label_name = 'Original',
                                                density = True,
                                                title=f'Column {column}',
                                                x_axis_label = 'Values', 
                                                y_axis_label = 'Frequencies')
            
                generate_distribution_histogram(dataframe = num_X_train2, 
                                                column_name = column, 
                                                number_bins = bins, 
                                                alpha = 0.3, 
                                                color = 'darkviolet', 
                                                edgecolor = 'black',
                                                label_name = 'Orig + Aug',
                                                density = True,
                                                title=f'Column {column}',
                                                x_axis_label = 'Values', 
                                                y_axis_label = 'Frequencies')

                plt.show()
        
                print(f"Column {column}:")
                mann_whitney_u_test(list(num_X_train1[column]), list(num_X_train2[column]))
            
    if categorical_df1.empty or categorical_df2.empty:
        print("Skipping categorical distribution comparison due to missing categorical data.\n")
    else:
        cat_X_train1 = categorical_df1.iloc[:, :-1] if categorical_df1.shape[1] > 1 else categorical_df1
        cat_X_train2 = categorical_df2.iloc[:, :-1] if categorical_df2.shape[1] > 1 else categorical_df2

        for column in cat_X_train1.columns.intersection(cat_X_train2.columns):
            value_counts1 = cat_X_train1[column].value_counts(normalize=True).sort_index()
            value_counts2 = cat_X_train2[column].value_counts(normalize=True).sort_index()

            all_categories = value_counts1.index.union(value_counts2.index)
            value_counts1 = value_counts1.reindex(all_categories, fill_value=0)
            value_counts2 = value_counts2.reindex(all_categories, fill_value=0)

            print(f"Column {column}:")
            print("df1:", value_counts1.values)
            print("df2:", value_counts2.values)

            L2_norm = np.linalg.norm(value_counts1.values - value_counts2.values, ord=2) / \
                      np.linalg.norm(value_counts1.values, ord=2)
            print("Euclidean norm of difference:", L2_norm)
            print()
            
    print("-------------Feature Distribution Comparisons-------------------- \n")

    feat_dist(numerical_df1, categorical_df1, numerical_df2, categorical_df2)
                    
    ############################## Correlation between columns (Feature vs Feature)  ##################################
    
    # Function to find correlation between numerical features
    def num_corr(X_train_numerical):
        if X_train_numerical.empty:  # Checking if X_train_numerical is empty
        print("Warning: Numerical dataframe is empty. Skipping correlation between numerical columns (FvF).")
        return None  # Return None if empty

        matrix = X_train_numerical.corr(method='pearson')
        print("---------------------------Correlation Matrix------------------------- \n", matrix)
        return matrix
    
    # Check if numerical_df1 and numerical_df2 are empty before proceeding
    if numerical_df1.empty or numerical_df2.empty:
        print("Error: One or both numerical datasets are empty. Correlation analysis skipped.")  # ✅ Error message
    else:
        # Correlation matrix for numerical data1
        correlation_matrix1 = num_corr(numerical_df1)
        correlation_df1 = pd.DataFrame(correlation_matrix1)
        if correlation_df1 is not None:
            print("\nCorrelation DataFrame 1:")
            print(correlation_df1)
            print(f"Shape: {correlation_df1.shape}")

        # Correlation matrix for numerical data2
        correlation_matrix2 = num_corr(numerical_df2)
        correlation_df2 = pd.DataFrame(correlation_matrix2)
        if correlation_df2 is not None:
            print("\nCorrelation DataFrame 2:")
            print(correlation_df2)
            print(f"Shape: {correlation_df2.shape}")
        # Checking correlation matrices exist before proceeding
        if correlation_df1 is not None and correlation_df2 is not None:
            # Convert the dataframes to numpy arrays
            matrix1 = correlation_df1.to_numpy()
            matrix2 = correlation_df2.to_numpy()

            # Compute the Frobenius norm of the difference between the matrices
            frobenius_abs = np.linalg.norm(matrix1 - matrix2, ord='fro')   # Absolute error with Frobenius norm
            frobenius_rel = frobenius_abs / np.linalg.norm(matrix1, ord='fro')    # Relative error with Frobenius norm

            print(f"Frobenius norm (absolute error): {frobenius_abs:.3f}")
            print(f"Frobenius norm (relative error): {frobenius_rel:.3f}")

    ################################# Chi-Square (Feature vs Feature)  ##########################################
    
    print("\n------------------Chi-Squared for Features v. Features-----------------------")
    # Function to find dependency between all categorical features
    def chi_squared_fvf(X_train_categorical):
        # Extract variable names
        variable_names = list(X_train_categorical.columns)

        # Initialize matrices to store chi-squared and p-values
        num_variables = len(variable_names)
        chi_squared = np.zeros((num_variables, num_variables))
        p_values = np.zeros((num_variables, num_variables))

        # Compute chi-squared and p-values for each pair of variables
        for i, j in combinations(range(num_variables), 2):
            contingency_table = pd.crosstab(X_train_categorical.iloc[:, i], X_train_categorical.iloc[:, j])
            
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

        # Printing the matrix-like output with variable names
        print("Chi-Squared Statistics:")
        print(chi_squared_df)
        print("\nP-Values:")
        print(p_values_df)

        return p_values_df

    if categorical_df1.empty or categorical_df2.empty:
        print("Error: One or both categorical datasets are empty. Skipping categorical analysis.")  # Error message
    else:
        # Chi-Square test for categorical data
        p_values_df1 = chi_squared_fvf(categorical_df1)
        p_values_df2 = chi_squared_fvf(categorical_df2)

        # Create a new DataFrame with True/False based on the p_value condition
        print("----------- Chi-Square (F vs F) True and False for Data1 ------------")
        p_value_df1 = p_values_df1 < 0.05
        print(p_value_df1)

        print("----------- Chi-Square (F vs F) True and False for Data2 ------------")
        p_value_df2 = p_values_df2 < 0.05
        print(p_value_df2)

        # Count the changes between the two DataFrames
        changes = (p_value_df1 != p_value_df2).sum().sum()

        # Display the number of changes
        print(f"Number of changes between data1 and data2 in Chi-Square (F vs F): {changes}")

    ################################### Chi-Square (Feature vs Label)  ##########################################
    # Extract y_train (label column)
    y_train1 = data1.iloc[:, 12]
    y_train2 = data2.iloc[:, 12]
    
    print("\n------------------------Chi-Square (F vs label column)------------------------")
    # Function to find dependency between all categorical features and the label
    def chi_squared_fvl(X_train_categorical, y_train):
        # Combining categorical X_train and y_train
        df = X_train_categorical.copy()
        df['label'] = y_train

        # Number of features, excluding label
        var_count = len(df.columns) - 1

        # Creates an empty array to print values in a table
        results = []

        for i in range(var_count):
            # Create contingency table of all features v. label
            crosstab = pd.crosstab(df.iloc[:, i], df.iloc[:, -1])
            
            # Compute chi-squared and p-values
            chi2 = stats.chi2_contingency(crosstab)[0]
            p = stats.chi2_contingency(crosstab)[1]
            
            # Append results to the list
            results.append({
                "Feature": df.columns[i],
                "Chi Squared Statistic": chi2,
                "P-Value": p
            })

        # Create a DataFrame from the results
        results_df = pd.DataFrame(results)

        # Print the DataFrame
        print("Label:", df.columns.values[-1])
        print(results_df.to_string(index=False))

        return results_df
    
    if categorical_df1.empty or categorical_df2.empty:
        print("Error: One or both categorical datasets are empty. Skipping categorical analysis.")  # ✅ Error message
    else:
        results_df1 = chi_squared_fvl(categorical_df1, y_train1)
        results_df2 = chi_squared_fvl(categorical_df2, y_train2)

        print("----------- Chi-Square (F vs L) True and False for Data1 ------------")
        p_value_fvl_df1 = results_df1['P-Value'] < 0.05
        print(p_value_fvl_df1)

        print("----------- Chi-Square (F vs L) True and False for Data2 ------------")
        p_value_fvl_df2 = results_df2['P-Value'] < 0.05
        print(p_value_fvl_df2)
        
        # Count the changes between the two DataFrames
        changes = (p_value_fvl_df1 != p_value_fvl_df2).sum()

        # Display the number of changes
        print(f"Number of changes between data1 and data2 in Chi-Square (F vs L): {changes}")
        
        
        
    ###########################################################################


