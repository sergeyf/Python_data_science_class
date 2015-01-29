# Let's find out how often James Joyce used each of the 26 letters
# in his novel "Ulysses", both in lower case and upper case

# Step 1: define the relevant dictionaries and initialize them
lowerKeys = 'abcdefghijklmnopqrstuvwxyz'
lowerDict = {}

# INSERT CODE INITIALIZING lowerDict HERE


upperKeys = lowerKeys.upper() # no need to retype it all
upperDict = {}

# INSERT CODE INITIALIZING upperDict HERE

# Step 1: importing the novel and updating our dictionaries
file_name = 'data/ulysses.txt'
f = open(file_name,'r')

for line in f: # what wonderful syntax!
    # INSERT CODE UPDATING THE DICTIONARIES HERE
     
f.close() # don't forget to close it!


# Step 2: it would be good to have normalized versions
# of each dictionary so we can compare
lowerDictNormed = lowerDict.copy()

# INSERT CODE TO NORMALIZE lowerDictNormed HERE
    
upperDictNormed = upperDict.copy()

# INSERT CODE TO NORMALIZE upperDictNormed HERE
    
# Step 3: print the values alongside each other using the 
for key in lowerKeys:
    print(key,lowerDictNormed[key],key.upper(),upperDictNormed[key.upper()])