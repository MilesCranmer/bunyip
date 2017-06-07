import bifrost as bf
import bifrost.pipeline as bfp
import bifrost.blocks as blocks
import bifrost.views as views

from copy import deepcopy

from .timeit import timeit

import numpy as np

class AccumulateBlock(bfp.TransformBlock):
    def __init__(self, iring, n_acc, *args, **kwargs):
    	super(AccumulateBlock, self).__init__(iring, *args, **kwargs)
        self.n_acc = n_acc
        self.acc_counter = 0
        
    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        # Tensor shape is (sequence, pol, coarse channel, time)
        tensor = ohdr['_tensor']
        shape = deepcopy(tensor['shape'])
        shape[0] = 1
        self.acc_data = np.zeros(shape, dtype='float32')
              
        return ohdr
    
    @timeit
    def on_data(self, ispan, ospan):
        self.acc_counter += 1
        print("ACC ID: %s of %s" % (self.acc_counter, self.n_acc))
        self.acc_data[:] += ispan.data
        
        if self.acc_counter % self.n_acc == 0:
            ospan.data[...] = self.acc_data
            self.acc_data[:] = 0
            self.acc_counter = 0
            return 1
        else:
            return 0

def accumulate(iring, n_acc, *args, **kwargs):
    return AccumulateBlock(iring, n_acc, *args, **kwargs)