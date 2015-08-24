import numpy as np

def step(a, b):
    print 'a: {}'.format(a)
    print 'b: {}'.format(b)
    return  [a+0.1] + b

# print step(1, 2, 3, 4, 5)

def bar(x, y):
    print 'x: {}'.format(x)
    print 'y: {}'.format(y)
    print np.sum(y)
    return step(x, y[0])

foo = lambda x, y: step(x, y[1:])

# print bar(1, [2, 3, 4, 5])
print foo(1, [2, 3, 4, 5])

l = [1]
print l[0:]
print l[1:]
