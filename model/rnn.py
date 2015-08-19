import theano
import numpy as np
import os
from theano import tensor as T
from collections import OrderedDict
from basenn import *
from theano_lstm import *

class RNN1L(BaseNN):
    """
    1-Layer RNN
    """
    
    def __init__(self, 
                    cells,
                    output_type='real',
                    dynamics=lambda x, y: x,
                    optimize_method='adadelta',
                    initial_lr=0.1,
                    initial_rho=0.95,
                    en_generate=True):
        '''
        Initialization function.

        Paramters
        ---------
        output_type: str
            Output type. 
        dynamics: y = function(x, y_)
            Dynamic Function of the output.
        optimize_method: str
            Optimization method: 'adagrad', 'adadelta', or 'sgd'. 
        initial_lr: float
            Initial learning rate.
        initial_rho: float
            Initial rho for adadelta.
        en_generate: bool
            If True, generative functions are enabled;
            if False, it's not. 
        '''
        self.cells = cells
        self.dynamics = dynamics
        self.output_type = output_type
        self.optimize_method = optimize_method
        self.en_generate = en_generate

        self.x = T.matrix(name='x') # input
        self.d = T.matrix(name='d') # groud truth output

        self.params = self.cells.params

        # recurrent function
        def _step(x_t, *hs_tm1):
            states_t = self.cells.forward(x_t, hs_tm1)
            return  [self.dynamics(x_t, states_t[-1])] + states_t[:-1]

        initial_y = T.zeros_like(self.x[0])

        results, _ = theano.scan(
            fn=lambda x_t, y_tm1, *hs_tm1: _step(x_t, *hs_tm1),
            sequences=self.x, 
            outputs_info=[initial_y] +\
                        [dict(initial=layer.initial_hidden_state, taps=[-1])\
                        for layer in self.cells.layers\
                        if hasattr(layer, 'initial_hidden_state')])
        self.y = results[0]
        self.hs = results[1:]

        # optimization
        if self.output_type == 'real':
            self.loss = T.mean((self.d - self.y) ** 2)
        else:
            raise Exception('Undefined output_type %s' % self.output_type)

        if self.optimize_method == 'adadelta':
            updates, gsums, xsums, self.lr, max_norm = \
                create_optimization_updates(self.loss, self.cells.params, 
                                            method=self.optimize_method,
                                            rho=initial_rho)
        else:
            updates, gsums, xsums, self.lr, max_norm = \
                create_optimization_updates(self.loss, self.cells.params, 
                                            method=self.optimize_method,
                                            lr=initial_lr)

        # interface
        self.train = theano.function(inputs=[self.x, self.d], 
                                        outputs=self.loss, 
                                        updates = updates, 
                                        allow_input_downcast=True)
        self.predict = theano.function(inputs=[self.x], 
                                        outputs=self.y)

        self.get_loss = theano.function(inputs=[self.x, self.d], 
                                        outputs=self.loss)

        # generation
        if self.en_generate:
            # Note: input and output should have the same dimension.
            self.n_gen = T.iscalar('n_gen') # number of steps to generate

            # generate the new sequence
            def _generate(seed):
                y_seed = seed[0]
                hs_seed = seed[1:]
                results_gen, _ = theano.scan(
                                    fn=_step,
                                    outputs_info=[y_seed] + hs_seed,
                                    n_steps=self.n_gen-1)
                return results_gen

            results_gen =\
                _generate([self.y[-1, :]] + [h[-1, :] for h in self.hs])
            self.y_gen =\
                T.concatenate([self.y[[-1]], results_gen[0]], axis=0)

            self.generate = theano.function(inputs=[self.x, self.n_gen], 
                                            outputs=self.y_gen)

class RNN1LZ(BaseNN):
    """
    1-Layer RNN with Input Z
    """
    
    def __init__(self, 
                    cells,
                    output_type='real',
                    dynamics=lambda x, y: x,
                    optimize_method='adadelta',
                    initial_lr=0.1,
                    initial_rho=0.95,
                    en_generate=True):
        '''
        Initialization function.

        Paramters
        ---------
        output_type: str
            Output type. 
        dynamics: y = function(x, y_)
            Dynamic Function of the output.
        optimize_method: str
            Optimization method: 'adagrad', 'adadelta', or 'sgd'. 
        initial_lr: float
            Initial learning rate.
        initial_rho: float
            Initial rho for adadelta.
        en_generate: bool
            If True, generative functions are enabled;
            if False, it's not. 
        '''
        self.cells = cells
        self.dynamics = dynamics
        self.output_type = output_type
        self.optimize_method = optimize_method
        self.en_generate = en_generate

        self.x = T.matrix(name='x') # input
        self.z = T.matrix(name='z') # input
        self.d = T.matrix(name='d') # groud truth output

        self.params = self.cells.params

        # recurrent function
        def _step(x_t, z_t, *hs_tm1):
            xz_t = T.concatenate([x_t, z_t])
            states_t = self.cells.forward(xz_t, hs_tm1)
            return  [self.dynamics(x_t, states_t[-1])] + states_t[:-1]

        initial_y = T.zeros_like(self.x[0])
        # T.zeros_like(T.concatenate([self.x[[0]], self.z[[0]]], axis=1))

        results, _ = theano.scan(
                fn=lambda x_t, z_t, y_tm1, *hs_tm1: _step(x_t, z_t, *hs_tm1),
                sequences=[self.x, self.z],
                outputs_info=[initial_y] +\
                        [dict(initial=layer.initial_hidden_state, taps=[-1])\
                        for layer in self.cells.layers\
                        if hasattr(layer, 'initial_hidden_state')])
        self.y = results[0]
        self.hs = results[1:]

        # optimization
        if self.output_type == 'real':
            self.loss = T.mean((self.d - self.y) ** 2)
        else:
            raise Exception('Undefined output_type %s' % self.output_type)

        if self.optimize_method == 'adadelta':
            updates, gsums, xsums, self.lr, max_norm = \
                create_optimization_updates(self.loss, self.cells.params, 
                                            method=self.optimize_method,
                                            rho=initial_rho)
        else:
            updates, gsums, xsums, self.lr, max_norm = \
                create_optimization_updates(self.loss, self.cells.params, 
                                            method=self.optimize_method,
                                            lr=initial_lr)

        # interface
        self.train = theano.function(inputs=[self.x, self.z, self.d], 
                                        outputs=self.loss, 
                                        updates = updates, 
                                        allow_input_downcast=True)
        self.predict = theano.function(inputs=[self.x, self.z], 
                                        outputs=self.y)

        self.get_loss = theano.function(inputs=[self.x, self.z, self.d], 
                                        outputs=self.loss)

        # generation
        if self.en_generate:
            # Note: input and output should have the same dimension.
            self.z_gen = self.z[self.x.shape[0]:, :]

            # generate the new sequence
            def _generate(seed):
                y_seed = seed[0]
                hs_seed = seed[1:]
                results_gen, _ = theano.scan(
                        sequences=self.z_gen,
                        fn=lambda z_t, y_t, *hs_tm1: _step(y_t, z_t, *hs_tm1),
                        outputs_info= [y_seed] + hs_seed)
                return results_gen

            results_gen =\
                _generate([self.y[-1, :]] + [h[-1, :] for h in self.hs])
            self.y_gen =\
                T.concatenate([self.y[[-1]], results_gen[0]], axis=0)

            self.generate = theano.function(inputs=[self.x, self.z],
                                            outputs=self.y_gen)

