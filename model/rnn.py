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
    
    def __init__(self, dim_x, dim_h, dim_y,
                    activation=T.nnet.sigmoid, output_type='real',
                    cumulative=True,
                    numpy_rng=None):
        '''
        dimh :: dimension of the hidden layer
        dimx :: dimension of the input
        dimy :: dimension of output
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


        [self.h, self.y], _ = theano.scan(fn=self.step,
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
    def step(self, x_t, h_tm1):
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
        _x_t = T.vector(name='x_t')
        _h_tm1 = T.vector(name='h_tm1')
        [_h_t, _y_t] = self.step(_x_t, _h_tm1)
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

        
        # def step_gen(h_tm1, y_tm1, _A):
        #     h_t = self.activation(T.dot(y_tm1, self.Wxh) +\
        #                             T.dot(h_tm1, self.Whh) + self.bh)
        #     y_t = T.dot(h_t, self.Why) + self.by
        #     return h_t, y_t

        # [h_gen, y_gen], _ = theano.scan(fn=step_gen,
        #                                     outputs_info=[self.h0, x_gen],
        #                                     non_sequences=[0],
        #                                     n_steps=n_gen)

        # if self.output_type == 'real':
        #     y_out_gen = y_gen
        # else:
        #     raise Exception('Undifined output_type: %s,', self.output_type)

        # f_y_gen = theano.function(inputs=[], outputs=y_gen)
        # f_h_gen = theano.function(inputs=[], outputs=h_gen)

        # return f_y_gen(), f_h_gen()


