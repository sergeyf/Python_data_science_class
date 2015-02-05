# Write some functions that make use of Pandas
# and answer questions about the Titanic dataset.

# First we import as before.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster") # makes thicker lines
plt.rcParams['figure.figsize'] = (10.0, 5.0)

# import the data
titanic = # INSERT STUFF HERE

# Function 1: returns the total number of passangers
# between to given ages: "age1" and "age2"
# where "age1" < "age2"
def totalPassangers(titanic,age1,age2):
    # INSERT STUFF HERE

# Function 2: prints out how many passangers 
# of each sex survived
def survivalBySex(titanic):
    # INSERT STUFF HERE

# Function 3: prints the correlations between 
# age and survived
# fare and survived
# pclass and survived
def survivalCorrelation(titanic):
    # INSERT STUFF HERE