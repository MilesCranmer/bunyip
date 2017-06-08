#!/usr/bin/env python

import glob, os
import numpy as np

import bifrost as bf
import bifrost.pipeline as bfp
import bifrost.blocks as blocks
import bifrost.views as views



class BinaryFileRead(object): 
    """ Simple file-like reading object for pipeline testing
    
    Args:
        filename (str): Name of file to open
        dtype (np.dtype or str): datatype of data, e.g. float32. This should be a *numpy* dtype,
                                 not a bifrost.ndarray dtype (eg. float32, not f32)
        gulp_size (int): How much data to read per gulp, (i.e. sub-array size)
    """
    def __init__(self, filename, gulp_size, dtype):
        super(BinaryFileRead, self).__init__()
        self.file_obj = open(filename, 'r')
        self.dtype = dtype
        self.gulp_size = gulp_size
        
    def read(self):
        d = np.fromfile(self.file_obj, dtype=self.dtype, count=self.gulp_size)
        return d
        
    def __enter__(self):
        return self
    
    def close(self):
        pass
    
    def __exit__(self, type, value, tb):
        self.close()


class MbBinaryFileReadBlock(bfp.SourceBlock):
    """ Block for reading binary data from file and streaming it into a bifrost pipeline
    
    Args:
        filenames (list): A list of filenames to open
        gulp_size (int): Number of elements in a gulp (i.e. sub-array size)
        gulp_nframe (int): Number of frames in a gulp. (Ask Ben / Miles for good explanation)
        dtype (bifrost dtype string): dtype, e.g. f32, cf32
    """
    def __init__(self, filenames, n_beams, fine_time_len, n_pol, gulp_nframe=1, *args, **kwargs):
        super(MbBinaryFileReadBlock, self).__init__(filenames, gulp_nframe, *args, **kwargs)
        self.dtype = 'cf32'
        self.gulp_size = n_beams * fine_time_len * n_pol
        print self.gulp_size
        
    def create_reader(self, filename):
        print "Loading %s" % filename
        np_dtype = 'complex64'
        
        return BinaryFileRead(filename, self.gulp_size, np_dtype)
         
    def on_sequence(self, ireader, filename):        
        ohdr = {'name': filename,
                '_tensor': {
                        'dtype':  self.dtype,
                        'shape':  [-1,  n_beams, fine_time_len, n_pol],
                        'labels': ['time', 'beam', 'fine_time', 'pol']
                        }, 
                }
        return [ohdr]
    
    def on_data(self, reader, ospans):
        indata = reader.read()
        if len(indata) == n_beams * fine_time_len * n_pol:
            indata = indata.reshape((1, n_beams, fine_time_len, n_pol))
            ospans[0].data[:] = indata
            return [1]
        else:
            return [0]

def gen_fake_data(n_blocks, n_beams, fine_time_len, n_pol, n_files):
    """ Generate some fake data that is kinda guppi raw like. No header, just data blob"""
    # Generate test data
    # NOTE: This make be several GB, depending on your choice of fine_time_len
    
    if not os.path.exists('testdata'):
        os.mkdir('testdata')
    
    print "Generating sine wave dataset"


    d = np.zeros((n_blocks, n_beams, fine_time_len, n_pol), dtype='complex64')
    t = np.arange(fine_time_len)
    for ii in range(1, n_files+1):
        w = 0.01 * ii
        s = np.sin(w*t, dtype='complex64')
        d[..., 0] = s
        d[..., 1] = s
        print "To file: ", d.shape
        d.tofile('testdata/mb_sin_data0%i.bin' % ii)
    print "Data generated."

if __name__ == "__main__":
    
    #######
    ## GENERATE FAKE DATA
    #######
    
    # Need to generate test data once only
    generate_test_data  = False
    fine_time_len       = 2**21     
    n_blocks            = 2
    n_pol               = 2
    n_beams             = 13
    n_files             = 8

    if generate_test_data:
        gen_fake_data(n_blocks, n_beams, fine_time_len, n_pol, n_files)

    ##########
    ## PIPELINE CONFIG
    ##########

    # Setup input data
    filenames   = sorted(glob.glob('testdata/mb_sin*.bin'))
    
    # Config for mid-resolution FFT
    n_fft_midres  = 16384
    n_int_midres  = 1024
    n_gulp_midres = 4
    
    # Config for low-resolution FFT
    n_fft_lowres  = 1024
    n_int_lowres  = 128
    n_gulp_lowres = 16
    
    
    #######
    ## PIPELINE DEFN
    #######
    
    # Read file into ring buffer
    b_read      = MbBinaryFileReadBlock(filenames, n_beams, fine_time_len, n_pol, gulp_nframe=1, buffer_nframe=4, core=0)
    blocks.print_header(b_read)
    
    
    # Reshape array for midres FFT
    with bf.block_scope(fuse=True, core=1):
        b_midres      = views.split_axis(b_read, axis='fine_time', n=n_fft_midres, label='fft_window') 
        b_midres      = blocks.transpose(b_midres, axes=['time', 'fine_time', 'beam', 'fft_window', 'pol'])   
        b_midres      = views.merge_axes(b_midres, 'time', 'fine_time')
        b_midres      = blocks.copy(b_midres, space='system', gulp_nframe=n_gulp_midres)
        #blocks.print_header(b_midres)
        
    # Do midres FFT + Detect on GPU
    with bf.block_scope(fuse=True, core=3):
        b_midres    = blocks.copy(b_midres,  space='cuda')
        b_midres    = blocks.FftBlock(b_midres, axes='fft_window', axis_labels='freq')
        b_midres    = blocks.fftshift(b_midres, axes='freq')
        b_midres    = blocks.detect(b_midres, mode='stokes')
        b_midres    = blocks.accumulate(b_midres, n_int_midres)
        b_midres    = blocks.copy(b_midres, space='system')
        
    # Reshape array for lowres FFT
    with bf.block_scope(fuse=True, core=1):
        b_lowres      = views.split_axis(b_read, axis='fine_time', n=n_fft_lowres, label='fft_window') 
        b_lowres      = blocks.transpose(b_lowres, axes=['time', 'fine_time', 'beam', 'fft_window', 'pol'])   
        b_lowres      = views.merge_axes(b_lowres, 'time', 'fine_time')
        b_lowres      = blocks.copy(b_lowres, space='system', gulp_nframe=n_gulp_lowres)
        blocks.print_header(b_lowres)
        
    # Do lowres FFT + Detect on GPU
    with bf.block_scope(fuse=True, core=3):
        b_lowres    = blocks.copy(b_lowres,  space='cuda')
        b_lowres    = blocks.FftBlock(b_lowres, axes='fft_window', axis_labels='freq')
        b_lowres    = blocks.fftshift(b_lowres, axes='freq')
        b_lowres    = blocks.detect(b_lowres, mode='stokes')
        b_lowres    = blocks.accumulate(b_lowres, n_int_lowres)
        b_lowres    = blocks.copy(b_lowres, space='system')

        
    # Run pipeline
    blocks.print_header(b_lowres)
    blocks.print_header(b_midres)
    pipeline = bfp.get_default_pipeline()
    print pipeline.dot_graph()
    pipeline.run()

