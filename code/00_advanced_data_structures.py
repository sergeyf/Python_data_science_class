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
D = {}

# beware potential confusion with sets
print( type({}) )
print( type({1,2}) )
print( type({1:2}) )

# dictionaries are ways to associate "keys" and "values"
D['this is a key'] = 'this is a value'
print(D['this is a key'])
D[1] = 2
print(D)

# we can get a list of all the keys
print(D.keys())

# and a list of all the values
print(D.values())

# what if the key doesn't exist?
print(D['dog'])

# we can get the value without worrying about errors
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