# tuples
t = [-1,1]
print( type(t) )
t = (-1,1)
print( type(t) )

# have to be a little careful:
c = [-1]
print( type(c) )

c = (-1)
print( type(c) )

c = ()
print( type(c) )

c = (-1,) # notice the comma!
print( type(c) )



# tuples are very similar to lists:
t = (-1,1,'infinity',1e5)
print(t[0])
print(t[-1])
print(t[1:])



# how are they different?
# can't modify them after assignment!
t[0] = 5
t.append(2)



# all we can do is concatenate tuples together:
print( c + t )






# sets
s1 = set([1,2,3,4,1])
print(s1) # only keeps one of each!

s2 = set('sam i am')
print(s2) # remember, strings are like lists of characters


# we can check if elements are in sets
print( 1 in s1 )

print( 'sam' in s2 )

print( 's' in s2 )


# we can add elements to sets
s1.add(5)
print(s1)

# but we can only add "hashable" elements (not lists)
s1.add([1,2])
# An object is "hashable" if it has a hash value 
# which never changes during its lifetime
# so: not lists, sets or dictionaries 

# tuples are hashable!
s1.add( ('dog','cat') )
print( s1 )


# we can add sets together
print(s1.union(s2))

# or get intersections (what elements are in common)
print(s1.intersection(s2))




# dictionaries!
D = {'one':1, 'two':2}  # note the colon!



# beware potential confusion with sets
print( type({}) )
print( type({1,2}) )
print( type({'one':1}) )



# dictionaries are ways to associate "keys" and "values"
D['this is a key'] = 'this is a value'
print(D['this is a key'])
D[1] = 2
print(D)



# we can get a list of all the keys
x = list(D.keys())
print( x ) 



# and a list of all the values
x = list(D.values())
print( x )



# we can also get back a list of tuples -> [(key,value)]
x = list(D.items())
print( x ) 



# what if the key doesn't exist?
print(D['dog'])



# we can get a value without worrying about errors
print(D.get('dog'))



# or put in a custom return value
print(D.get('dog','No such key!'))



# just like D.get(), but sets the value for us as well
print(D.setdefault('dog','cat'))



# we can combine two dictionaries with the update method
E = {'dog':'canine',2:3}
print(D)
D.update(E) # this happens in place
print(D) # some of D has been overwritten by E!



# 3 different ways to populate a dictionary!

# number 1 - curly braces & colons
d = {"favorite cat": None,
     "favorite spam": "all"}
print(d)


# number 2 - just start filling in items/keys
d = {}  # empty dictionary
d['cat'] = 'dog'
d['one'] = 1
d['two'] = 2
print(d)


# number 3 - a list of tuples as a dictionary
mylist = [("cat","dog"), ("one",1),("two",2)]
print( dict(mylist) == d )






# dictionaries have no ordering!
print(d[0]) # this doesn't work because 0 is not a key



# you can put anything in a dictionary!
D = {}
D['a list'] = [0,1,2,3]
D['a dict'] = {'one':1, 'two':2}
print(D)

# now we can reference sub-dictionaries like this:
print( D['a dict']['one'] )



# deleting items:
del D['a list']
print(D)






# finally - list comprehensions (by example)

# Example: imagine you want a list of all numbers 
# from 0 to 100 which are divisible by 7 or 11

# normal way to do it:
L = [] # initialize list
for i in range(100):
    if (i % 7 == 0) or (i % 11 == 0):
        L.append(i)
        
print(L)

# with list comprehensions:
L = [i for i in range(100) if (i % 7 == 0) or (i % 11 == 0)]




# let's write a function that prints out 
# the keys and values of a dictionary
# but sorted by key or value, as the user specifies
# you'll need to use the "sorted()" function!

# test input:
D = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5}

# we will need a list of item tuples because 
# sorted(D) only returns the keys
items = list(D.items())

# it's easy to sort by the keys:
print( sorted(items) )



# but how to sort by the values? 
# let's look at the "sorted" doc

# ...













# we need to sort by the second element!
# so let's write a function that does that:
def getSecondElement(x):
    return x[1]

# and now we can provide that function as our "key"
print( sorted(items, key=getSecondElement) ) 



# here is a more compact version, with a lambda function:
print( sorted(items, key=lambda x: x[1] ))




# lambda functions are for quick, simple functions:
getSecondElementLambda = lambda x: x[1] 

# this does the same thing as "getSecondElement"
# look up lambda functions later