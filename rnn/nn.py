import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class NNL1(object):
    
    def __init__(self, dimx, dimy, dimh):
        '''
        dimh :: dimension of the hidden layer
        dimx :: dimension of the input
        dimy :: dimension of output
        '''
        # parameters of the model
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimx, dimh)).astype(theano.config.floatX))
        self.Wgl  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimx, dimh)).astype(theano.config.floatX))
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimh, dimy)).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(dimh, dtype=theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(dimy, dtype=theano.config.floatX))

        # bundle
        self.params = [self.Wx, self.Wgl, self.W, self.bh, self.b]
        self.names  = ['Wx', 'Wgl', 'W', 'bh', 'b', 'h0']

        x = T.fmatrix(name='x') 
        gl = T.fvector(name='gl') # goal state
        d = T.fmatrix(name='d') 

        '''
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(gl, self.Wgl) +\
                                    T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.dot(h_t, self.W) + self.b + x_t
            # s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,\
                                sequences=x, outputs_info=[self.h0, None],\
                                n_steps=x.shape[0])
        '''
        h = T.nnet.sigmoid(T.dot(x, self.Wx) + T.dot(gl, self.Wgl) + self.bh)
        y = T.dot(h, self.W) + self.b + numpy.random.randn(dimy)

        cost = T.mean((x+y - d) ** 2)

        # cost and gradients and learning rate
        lr = T.scalar('lr') # learning rate
        # lr = theano.shared(numpy.array(lr).astype("float32"))
        gradients = T.grad( cost, self.params )
        updates = OrderedDict(( p, p-lr*g )\
                                for p, g in zip( self.params , gradients))
        
        # theano functions
        self.predict = theano.function(inputs=[x, gl], outputs=y)

        self.train = theano.function( inputs  = [x, gl, d, lr],
                                      outputs = cost,
                                      updates = updates )

        self.getCost = theano.function(inputs = [x, gl,  d], outputs = cost)

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), \
              newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), \
              borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value(borrow=False).flatten() \
                                    for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(numpy.load(filename))



