def add1(x): return x+1

print map(add1, [1,2,3])

def isOdd(x): return x%2 == 1
print filter(isOdd, [1,2,3,4])

def add(x,y): return x+y
print reduce(add, range(5))

print reduce(lambda x,y: x+y, filter(isOdd, map(lambda t: t[0], [(1,2),(2,4),(5,3)])))