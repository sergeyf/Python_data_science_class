# pandas! for dealing with real data
# from here:
# https://data.seattle.gov/Transportation/Fremont-Bridge-Hourly-Bicycle-Counts-by-Month-Octo/65db-xm6k
# 
# to get a sense of why pandas is so great
# let's try to import data by hand first...

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster") # makes thicker lines
plt.rcParams['figure.figsize'] = (10.0, 5.0)
import pandas as pd

# import the data
hourly = pd.read_csv("data/fremont_bridge_data.csv", index_col='Date', parse_dates=True)

# name the columns
hourly.columns = ['northbound', 'southbound']

# let's add a new column that is useful
hourly['total'] = hourly['northbound'] + hourly['southbound']

# resample the data to days and weeks
daily = hourly.resample('d','sum')

weekly = hourly.resample('w','sum')



# plot some data
plt.plot(daily)
plt.legend(['nb','sb','total'])
plt.show()



# too noisy! let's smooth it out
dailySmoothed = pd.stats.moments.rolling_mean(daily,window=30)
plt.plot(dailySmoothed)
plt.legend(['nb','sb','total'])
plt.show()



# pandas is HUGE -> pd.stats.moments.rolling_mean
# is 4 layers deep!




# let's stop showing off and do some basics

# panda has its own datatypes
print( type(daily) )
print( type(daily['total']) )


# we can make panda data from scratch
ssn = ['234-65-9080','167-65-9080','987-65-9080','456-65-9080','234-65-9080']
first = ['g.','y.','f.','w.','p.']
last = ['arumbo','gralph','ungde','uyino','phin']
age = [56,12,37,22,101]


# aside: the zip function makes tuples (built-in)
z = list( zip(first,last,age) )
print( z )


# now we can make a dataframe:
df = pd.DataFrame(index = ssn, data = z, columns=['first', 'last','age'])
print(df)
print(df.info())


# let's sort it by last name
df = df.sort(['last'])
print(df)


# oldest age?
maxAge = df['age'].max()
print(maxAge)


# oldest person?
print( df[df['age'] == maxAge] )



# ok now let's get back to your serious dataset
print( hourly.info() )

# print only the top of the dataset
print( weekly.head() )

# we can get only the index as a time series
print( weekly.index )

# we can print the values as a numpy array
print( weekly.values )

# how to reference a column:
print( weekly['northbound'] )

# how to reference multiple columns
print( weekly[['northbound','southbound']] )

# how to reference a row by index key
print( weekly.loc['2012-12-30'] )

# how to reference multiple rows by index key
print( weekly.loc[[weekly.index[0], weekly.index[1]],:] )
print( weekly.loc[pd.to_datetime(['2012-12-30','2012-12-30'])] )

# how to referene a row by numerical index
print( weekly.iloc[12] )

# how to reference a specific (row,column)
print( weekly['northbound']['2012-12-30'] )
print( weekly.loc['2012-12-30'] ['northbound'] )
print( weekly.loc['2012-12-30','northbound'] )

# we can also do this:
print( weekly['2012'] )
print( weekly['2012-12'] )

# but not this:
print( weekly['2013-12-30'] )
# for more info, you can try here:
# http://pandas.pydata.org/pandas-docs/stable/timeseries.html#datetimeindex-partial-string-indexing






# onwards!

# transform entire columns with any numpy function
print( np.log(weekly['southbound']) ) 


# we can also do numpy style indexing
print( weekly['southbound'][ weekly['southbound'] > 14000 ] )

# finding nulls
print( hourly[hourly['southbound'].isnull()] )

# getting a sense for the data
print( hourly.describe() )

# correlations and covariances
print( hourly.corr() )
print( hourly.cov() )





# further learning will happen on another dataset:
'''
These data are from a multicenter, 
randomized controlled trial of 
botulinum toxin type B (BotB) 
in patients with cervical dystonia from nine U.S. sites.

Randomized to placebo (N=36), 
5000 units of BotB (N=36), 
10,000 units of BotB (N=37)
Response variable: total score on 
Toronto Western Spasmodic Torticollis Rating Scale (TWSTRS), 
measuring severity, pain, and disability of cervical 
dystonia (high scores mean more impairment)
TWSTRS measured at baseline (week 0) 
and weeks 2, 4, 8, 12, 16 after treatment began
'''
cdystonia = pd.read_csv("data/cdystonia.csv", index_col=None)

# we can look at one patient at a time like this:
print( cdystonia.stack().head() )



# we can set the index to be composed of two columns:
# patient and observation number
cdystonia2 = cdystonia.set_index(['patient','obs'])
print( cdystonia2.head() )



# here is how to add a column that maps treatments strings
# to identifying integers
print( cdystonia['treat'].value_counts() )

# here is how to do that:
treatment_map = {'Placebo': 0, '5000U': 1, '10000U': 2}
cdystonia['treatment'] = cdystonia['treat'].map(treatment_map)
print( cdystonia['treatment'].value_counts() )



# we can also just replace the original column 
# using this method:
print(cdystonia2.head())
cdystonia2['treat'] = cdystonia2['treat'].replace(treatment_map)
print(cdystonia2.head())




# aggregation, slicing, transformation
cdystonia_grouped = cdystonia.groupby(cdystonia['patient'])
for patient, group in cdystonia_grouped:
    print("Patient ID is:",patient)
    print(group)
    print()
    


# here is how to get a specific group:
print( cdystonia_grouped.get_group(4) )

 
   
# return median of each column for each group
print( cdystonia_grouped.agg(np.median).head() )
# Note that the "treat" and "sex" variables missing.
# Can't take means of strings!


# some agg functions are so common, they are built-in:
print( cdystonia_grouped.mean().head() )

# and we should probably change the column names...
print( cdystonia_grouped.mean().add_suffix('_mean').head() )



# we can aggregate by multiple keys:
print( cdystonia.groupby(['treat','week']).mean()['twstrs'] )



# we can get a subset of the data
print( cdystonia.query("(obs == 6) & (age < 30) & (sex == 'F')" ) )



# scatter plots and histogram magic
pd.scatter_matrix(cdystonia[['age','twstrs']], diagonal='kde', figsize=(10, 10));
