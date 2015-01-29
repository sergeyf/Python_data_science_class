# Let's find out how often James Joyce used each of the 26 letters
# in his novel "Ulysses", both in lower case and upper case

# Step 1: define the relevant dictionaries and initialize them
lowerKeys = 'abcdefghijklmnopqrstuvwxyz'
lowerDict = {}
for key in lowerKeys:
    lowerDict[key] = 0 # initialize at 0 count

upperKeys = lowerKeys.upper() # no need to retype it all
upperDict = {}
for key in upperKeys:
    upperDict[key] = 0
        

# Step 1: importing the novel and updating our dictionaries
file_name = 'data/ulysses.txt'
f = open(file_name,'r')

for line in f: # what wonderful syntax!
    for letter in line:
        if letter in lowerKeys:
            lowerDict[letter] += 1
        elif letter in upperKeys:
            upperDict[letter] += 1
            
f.close() # don't forget to close it!


# Step 2: it would be good to have normalized versions
# of each dictionary so we can compare
lowerDictNormed = lowerDict.copy()
lowerTotal = sum(lowerDict.values()) # total number of letters
for key in lowerDictNormed.keys():
    lowerDictNormed[key] /= lowerTotal # dividing by the total
    
upperDictNormed = upperDict.copy()
upperTotal = sum(upperDict.values()) # total number of letters
for key in upperDictNormed.keys():
    upperDictNormed[key] /= upperTotal # dividing by the total
    
# Step 3: print the values alongside each other using the 
for key in lowerKeys:
    print(key,lowerDictNormed[key],key.upper(),upperDictNormed[key.upper()])