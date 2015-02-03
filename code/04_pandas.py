# pandas! for dealing with real data
# from here:
# https://data.seattle.gov/Transportation/Fremont-Bridge-Hourly-Bicycle-Counts-by-Month-Octo/65db-xm6k
# 
# to get a sense of why pandas is so great
# let's try to import data by hand first...

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

# how to referene a row by numerical index
print( weekly.iloc[12] )

# how to reference a specific (row,column)
print( weekly['northbound']['2012-12-30'] )
print( weekly.loc['2012-12-30'] ['northbound'] )
print( weekly.loc['2012-12-30','northbound'] )

