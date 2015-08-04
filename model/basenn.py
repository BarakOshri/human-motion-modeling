import theano
import numpy as np
import os
from theano import tensor as T
from collections import OrderedDict

class BaseNN(object):
    def __init__(self, num_rng=None):
        if not numpy_rng:
            self.numpy_rng = np.random.RandomState(1)
        else:
            self.numpy_rng = numpy_rng
        return

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = np.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), \
              newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), \
              borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return np.concatenate([p.get_value(borrow=False).flatten() \
                                    for p in self.params])

    def save(self, filename):
        np.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(np.load(filename))

    def init_param(self, size, mode='n', scale=.01):
        """
        Generate initial values of the parameters.

        Params
        ------
        mode: str 
                'normal' or 'n' for drawing from normal distribution, 
                'uniform' or 'u' for drawing from uniform distribution, 
                'log-uniform' or 'lu' for drawing from log uniform 
                distribution, 
                'repetitive' or 'r' for repeating same values in each element. 
        """
        if mode == 'normal' or mode == 'n':
            param_init = scale * self.numpy_rng.normal(size=size)\
                            .astype(theano.config.floatX)
        elif mode == 'uniform' or mode == 'u':
            if np.size(scale) == 1:
                low = -scale
                high = scale
            elif np.size(scale) == 2:
                low = scale[0]
                high = scale[1]
            param_init = self.numpy_rng.uniform(size=size, low=low, high=high)\
                        .astype(theano.config.floatX)
        elif mode == 'log-uniform' or mode == 'lu':
            param_init = np.exp(self.numpy_rng.uniform(
                                    low=scale[0], high=scale[1], size=size).\
                                    astype(theano.config.floatX))
        elif mode == 'repetitive' or mode == 'r':
            param_init = scale*np.ones(size,
                            dtype=theano.config.floatX)
        else:
            raise Exception('\''+str(mode)+'\'' + ' is not a valid mode. ')
        return param_init 



