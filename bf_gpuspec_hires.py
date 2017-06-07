import glob
import numpy as np

import bifrost as bf
import bifrost.pipeline as bfp
import bifrost.blocks as blocks
import bifrost.views as views

from copy import deepcopy

import blocks as byip

def timeit(method):
    """ Decorator for timing execution of a method in a class """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        classname  = repr(args[0]).split(' ')[0][10:]
        methodname = str(method.__name__)
        time_str   = '%2.2fs' % (te-ts)

        print '%24s %16s %16s' % (classname, methodname, time_str)
        return result
    return timed

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

with bfp.Pipeline() as pipeline:
    import sys
    import os
    try:
        filepath = sys.argv[1]
    except:
        print("Usage: ./gpuspec.py path_to_raw_files")
        exit()


    n_blocks = 128              # Number of blocks in raw file
    n_coarse = 64
    n_t_per_coarse = 524288
    n_pol    = 4                
    n_int    = 16               # Runtime - Number of integrations
    n_chunks = 2                # Runtime - Number of time chunks 
    n_t      = n_blocks / n_int / n_chunks
    n_chan   = n_coarse * n_t_per_coarse * n_chunks
    
    filelist = sorted(glob.glob(os.path.join(filepath, '*.raw')))
    
    # Read from guppi raw file
    b_guppi   = blocks.read_guppi_raw(filelist, core=1, buffer_nframe=4)
    b_gup2    = views.rename_axis(b_guppi, 'freq', 'channel')
    
    
    # Buffer up two blocks & reshape to allow longer FFT
    with bf.block_scope(fuse=True, core=2):
        b_gup2    = views.split_axis(b_gup2, axis='time', n=n_chunks, label='time_chunk')
        b_gup2    = blocks.transpose(b_gup2, axes=['time', 'channel', 'time_chunk', 'fine_time', 'pol'])
        b_gup2    = views.reverse_scale(b_gup2, 'time_chunk')
        b_gup2    = views.merge_axes(b_gup2, 'time_chunk', 'fine_time', label='fine_time')
        blocks.print_header(b_gup2)
        
        #blocks.print_header(b_gup2)
    
    # blocks.print_header(b_gup2)
   
    # Copy over to GPU and FFT
    with bf.block_scope(fuse=True, core=3):
        b_copy    = blocks.copy(b_gup2,  space='cuda')
        b_fft     = blocks.fft(b_copy,  axes='fine_time', axis_labels='freq')
        b_ffs     = blocks.fftshift(b_fft, axes='freq')
        #blocks.print_header(b_fft)
        b_pow     = blocks.detect(b_ffs, mode='stokes')
        b_copp    = blocks.copy(b_pow, space='system')
    
    #blocks.print_header(b_copp)
    # Flatten channel/freq axis to form output spectra and accumulate
    b_copp    = blocks.copy(b_copp, space='system', buffer_nframe=4)
    b_flat    = views.merge_axes(b_copp, 'channel', 'freq', label='freq')
    b_acc     = byip.accumulate(b_flat, n_int, core=4, buffer_nframe=4)
    #blocks.print_header(b_acc)
    b_hdf     = byip.hdf_writer(b_acc, shape=[n_t, n_chan, n_pol], dtype='float32', core=5)
    pipeline.run()
