import theano
import numpy as np
import os
from theano import tensor as T
from collections import OrderedDict
from basenn import *
from theano_lstm import *

class LSTML1(BaseNN):
    """
    1-Layer LSTM
    """
    
    def __init__(self, n_x, n_h, n_y,
                    activation=T.tanh, 
                    output_type='real',
                    dynamics=lambda x, y: x+y,
                    en_generate=True,
                    numpy_rng=None):
        '''
        Initialization function.

        Paramters
        ---------
        n_x: int
            Dimension of the input layer.
        n_h: int
            Dimension of the hideen layer.
        n_y: int
            Dimension of the output layer.
        activation: theano.tensor.elemwise.Elemwise
            Activation function.
        output_type: str
            Output type. 
        dynamics:

        en_generate: bool
            If True, generative functions are enabled
            if False, it's not. 
        numpy_rng: 
            Numpy random number generator.  
        '''
        self.activation = activation
        self.output_type = output_type
        self.dynamics = dynamics
        self.dropout = \
            theano.shared(np.float64(0.3).astype(theano.config.floatX))

        self.x = T.matrix(name='x') # input
        self.d = T.matrix(name='d') # groud truth output
        self.lr = T.scalar('lr') # learning rate

        self.rnn = StackedCells(n_x, layers=[n_h], 
                                activation=activation, celltype=LSTM)
        self.rnn.layers.append(Layer(n_h, n_y, lambda x: x))

        self.params = self.rnn.params

        def _step(x_t, *hs_tm1):
            states_t = self.rnn.forward(x_t, hs_tm1)
            return  states_t[:-1] + [dynamics(x_t, states_t[-1])]

        # def _gen_step(x_t, *h_tm1):
        #     s_t = self.rnn.forward(x_t, h_tm1)
        #     return  [dynamics(x_t, s_t[-1])] + s_t[:-1]

        results, _ = theano.scan(fn=_step,
            sequences=self.x, 
            outputs_info=[dict(initial=layer.initial_hidden_state, taps=[-1]) 
                            for layer in self.rnn.layers 
                            if hasattr(layer, 'initial_hidden_state')] + [None])
        self.hs = results[:-1]
        self.y = results[-1]

        self.loss = T.mean((self.d - self.y) ** 2)
        self.predict = theano.function(inputs=[self.x], 
                                        outputs=self.y)

        updates, gsums, xsums, lr, max_norm = \
            create_optimization_updates(self.loss, 
                                        self.rnn.params, 
                                        method='adadelta')

        self.train = theano.function(inputs=[self.x, self.d], 
                                        outputs=self.loss, 
                                        updates = updates, 
                                        allow_input_downcast=True)
        self.predict = theano.function(inputs=[self.x], 
                                        outputs=self.y)

        self.get_loss = theano.function(inputs=[self.x, self.d], 
                                        outputs=self.loss)

        if en_generate:
            # Note: input and output should have the same dimension.
            self.n_gen = T.iscalar('n_gen') # number of steps to generate

            # debug
            self.debug = theano.function(inputs=[self.x], 
                                            outputs=[self.y] + self.hs)

            # generate the new sequence
            def _generate(seed):
                y_seed = seed[0]
                hs_seed = seed[1:]
                results_gen, _ = theano.scan(
                                    fn=lambda hs_tm1, x_t: _step(x_t, hs_tm1),
                                    outputs_info= hs_seed + [y_seed],
                                    n_steps=self.n_gen-1)
                return results_gen

            results_gen =\
                _generate([self.y[-1, :]] + [h[-1, :] for h in self.hs])
            self.y_gen =\
                T.concatenate([self.y[[-1]], results_gen[-1]], axis=0)

            self.generate = theano.function(inputs=[self.x, self.n_gen], 
                                            outputs=self.y_gen)
