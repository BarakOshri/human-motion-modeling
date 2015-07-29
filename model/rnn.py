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


class RNNL1(BaseNN):
    """
    1-Layer Recurrent Neural Network
    """
    
    def __init__(self, dim_x, dim_h, dim_y,
                    activation=T.nnet.sigmoid, output_type='real',
                    cumulative=True,
                    numpy_rng=None):
        '''
        Initialization function.

        Paramters
        ---------
        dimx: int
            Dimension of the input layer.
        dimh: int
            Dimension of the hideen layer.
        dimy: int
            Dimension of the output layer.
        activation: theano.tensor.elemwise.Elemwise
            Activation function.
        output_type: str
            Output type. 
        cumulative: bool
            If True, the output is cumulative to previous output;
            if False, it's not. 
        numpy_rng: 
            Numpy random number generator.  
        '''
        # parameters
        if not numpy_rng:
            self.numpy_rng = np.random.RandomState(1)
        else:
            self.numpy_rng = numpy_rng

        Wxh_init = self.init_param((dim_x, dim_h), 'u', 0.2)
        self.Wxh = theano.shared(value=Wxh_init, name='Wxh')

        Whh_init = self.init_param((dim_h, dim_h), 'u', 0.2)
        self.Whh = theano.shared(value=Whh_init, name='Whh')

        Why_init = self.init_param((dim_h, dim_y), 'u', 0.2)
        self.Why = theano.shared(value=Why_init, name='Why')

        bh_init = self.init_param((dim_h,), 'r', 0.)
        self.bh = theano.shared(value=bh_init, name='bh')

        by_init = self.init_param((dim_y,), 'r', 0.)
        self.by = theano.shared(value=by_init, name='by')

        h0_init = self.init_param((dim_h,), 'r', 0.)
        self.h0 = theano.shared(value=h0_init, name='h0')

        self.params = [self.Wxh, self.Whh, self.Why, self.bh, self.by, self.h0 ]

        self.activation = activation
        self.output_type = output_type
        self.cumulative = cumulative

        self.x = T.matrix(name='x') # input
        self.d = T.matrix(name='d') # groud truth output
        self.lr = T.scalar('lr') # learning rate


        [self.h, self.y], _ = theano.scan(fn=self._step,
                                            sequences=self.x, 
                                            outputs_info=[self.h0, None])

        # loss, predict and generate
        if self.output_type == 'real':
            self.loss = T.mean((self.d - self.y) ** 2)
            self.predict = theano.function(inputs=[self.x], 
                                            outputs=self.y)
        else:
            raise Exception('Undifined output_type: %s,', self.output_type)

        # gradients and learning rate
        grads = T.grad(self.loss, self.params)
        updates = OrderedDict((p, p - self.lr*g)\
                                for p, g in zip( self.params , grads))
        
        # training interfaces
        self.train = theano.function(inputs=[self.x, self.d, self.lr],
                                      outputs=self.loss,
                                      updates=updates)
        self.get_loss = theano.function(inputs=[self.x, self.d], 
                                        outputs=self.loss)
    def _step(self, x_t, h_tm1):
        """
        Recurrent function
        """
        h_t = self.activation(T.dot(x_t, self.Wxh) +\
                                T.dot(h_tm1, self.Whh) + self.bh)
        if self.cumulative:
            y_t = T.dot(h_t, self.Why) + self.by + x_t
        else:
            y_t = T.dot(h_t, self.Why) + self.by
        return h_t, y_t

    def generate(self, x_seed, n_gen=10):
        """
        Genereate seqences of length of n_gen.
        """
        _x_t = T.vector(name='x_t')
        _h_tm1 = T.vector(name='h_tm1')
        [_h_t, _y_t] = self._step(_x_t, _h_tm1)
        _step = theano.function(inputs=[_x_t, _h_tm1], outputs=[_h_t, _y_t])

        y_t = np.void
        h_t = self.h0.get_value()
        for t in range(x_seed.shape[0]):
            h_t, y_t = _step(x_seed[t, :], h_t)

        y_gen = np.empty((n_gen, x_seed.shape[1]))
        y_gen[0, :] = y_t
        for t in range(n_gen-1):
            h_t, y_t = _step(y_t, h_t)
            y_gen[t+1, :] = y_t

        return y_gen


class GRNNL1(BaseNN):
    """
    Goal-Oriented 1-Layer Recurrent Neural Network
    """
    
    def __init__(self, dim_x, dim_h, dim_y,
                    activation=T.nnet.sigmoid, output_type='real',
                    cumulative=True,
                    numpy_rng=None):
        '''
        Initialization function.

        Paramters
        ---------
        dimx: int
            Dimension of the input layer.
        dimh: int
            Dimension of the hideen layer.
        dimy: int
            Dimension of the output layer.
        activation: theano.tensor.elemwise.Elemwise
            Activation function.
        output_type: str
            Output type. 
        cumulative: bool
            If True, the output is cumulative to previous output;
            if False, it's not. 
        numpy_rng: 
            Numpy random number generator.  
        '''
        # parameters
        if not numpy_rng:
            self.numpy_rng = np.random.RandomState(1)
        else:
            self.numpy_rng = numpy_rng

        Wgh_init = self.init_param((dim_x, dim_h), 'u', 0.2)
        self.Wgh = theano.shared(value=Wgh_init, name='Wgh')

        Wxh_init = self.init_param((dim_x, dim_h), 'u', 0.2)
        self.Wxh = theano.shared(value=Wxh_init, name='Wxh')

        Whh_init = self.init_param((dim_h, dim_h), 'u', 0.2)
        self.Whh = theano.shared(value=Whh_init, name='Whh')

        Why_init = self.init_param((dim_h, dim_y), 'u', 0.2)
        self.Why = theano.shared(value=Why_init, name='Why')

        bh_init = self.init_param((dim_h,), 'r', 0.)
        self.bh = theano.shared(value=bh_init, name='bh')

        by_init = self.init_param((dim_y,), 'r', 0.)
        self.by = theano.shared(value=by_init, name='by')

        h0_init = self.init_param((dim_h,), 'r', 0.)
        self.h0 = theano.shared(value=h0_init, name='h0')

        self.params = [self.Wxh, self.Wgh, self.Whh, self.Why, 
                        self.bh, self.by, self.h0 ]

        self.activation = activation
        self.output_type = output_type
        self.cumulative = cumulative

        self.g = T.vector(name='g') # goal
        self.x = T.matrix(name='x') # input
        self.d = T.matrix(name='d') # groud truth output
        self.lr = T.scalar('lr') # learning rate


        [self.h, self.y], _ = theano.scan(fn=self._step,
                                            sequences=self.x, 
                                            outputs_info=[self.h0, None],
                                            non_sequences=self.g)

        # loss, predict and generate
        if self.output_type == 'real':
            self.loss = T.mean((self.d - self.y) ** 2)
            self.predict = theano.function(inputs=[self.g, self.x], 
                                            outputs=self.y)
        else:
            raise Exception('Undifined output_type: %s,', self.output_type)

        # gradients and learning rate
        grads = T.grad(self.loss, self.params)
        updates = OrderedDict((p, p - self.lr*g)\
                                for p, g in zip( self.params , grads))
        
        # training interfaces
        self.train = theano.function(inputs=[self.g, self.x, self.d, self.lr],
                                      outputs=self.loss,
                                      updates=updates)
        self.get_loss = theano.function(inputs=[self.g, self.x, self.d], 
                                        outputs=self.loss)
    def _step(self, x_t, h_tm1, g):
        """
        Recurrent function
        """
        h_t = self.activation(T.dot(x_t, self.Wxh) +\
                                T.dot(g, self.Wgh) +\
                                T.dot(h_tm1, self.Whh) + self.bh)
        if self.cumulative:
            y_t = T.dot(h_t, self.Why) + self.by + x_t
        else:
            y_t = T.dot(h_t, self.Why) + self.by
        return h_t, y_t

    def generate(self, g, x_seed, n_gen=10):
        """
        Genereate seqences of length of n_gen.
        """
        _g = T.vector(name='_g') # goal
        _x_t = T.vector(name='_x_t')
        _h_tm1 = T.vector(name='_h_tm1')
        [_h_t, _y_t] = self._step(_x_t, _h_tm1, _g)
        _step = theano.function(inputs=[_x_t, _h_tm1, _g], outputs=[_h_t, _y_t])

        y_t = np.void
        h_t = self.h0.get_value()
        for t in range(x_seed.shape[0]):
            h_t, y_t = _step(x_seed[t, :], h_t, g)

        y_gen = np.empty((n_gen, x_seed.shape[1]))
        y_gen[0, :] = y_t
        for t in range(n_gen-1):
            h_t, y_t = _step(y_t, h_t, g)
            y_gen[t+1, :] = y_t

        return y_gen
