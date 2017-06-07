import bifrost as bf
import bifrost.pipeline as bfp
import bifrost.blocks as blocks
import bifrost.views as views

from copy import deepcopy

import h5py

from .timeit import timeit

class HdfWriteBlock(bfp.SinkBlock):
    def __init__(self, iring, shape, dtype, file_ext='hires.hdf', *args, **kwargs):
        super(HdfWriteBlock, self).__init__(iring, *args, **kwargs)
        self.current_h5 = None
        self.file_ext = file_ext
        self.idx = 0
        self.shape = shape
        self.dtype = dtype
    
    def on_sequence(self, iseq):
        if self.current_h5 is not None:
            self.current_h5.close()
            self.idx = 0
            
        new_filename = iseq.header['name'] + '.' + self.file_ext
        self.current_h5 = h5py.File(new_filename, 'w')
        
        print("Creating %s" % new_filename)
        
        self.data = self.current_h5.create_dataset("data", 
                                            shape=self.shape,
                                            dtype=self.dtype)
    
    @timeit
    def on_data(self, ispan):
        self.data[self.idx] = ispan.data
        self.idx += 1

def hdf_writer(iring, shape, dtype, file_ext='.hdf', *args, **kwargs):
    return HdfWriteBlock(iring, shape, dtype, file_ext='.hdf', *args, **kwargs)
