#!/usr/bin/env python
"""
# bf_gpuspec.py

GPU data reduction pipeline for Breakthrough Listen.

This script produces spectral data products for BL observations from 'guppi raw' files.
It computes three different combinations of (time, frequency) resolutions:

* High time, low frequency resolution
* Medium resolution product
* High frequency, low time resolution product

Of these, a mojority of GPU RAM is used by the high time resolution product.

# GPUSPEC equivalent

```
BASENAME=guppi_12345_23456_TARGET numactl --cpunodebind 1 /usr/local/listen/bin/gpuspec2 -i 
    ${BASENAME}.[0-9][0-9][0-9][0-9].raw -b 8 -B 2 
    -f 1048576,8,1024 -t 51,128,3072 
    -o ${BASENAME}.gpuspec.
```

"""

import glob
import numpy as np

import bifrost as bf
import bifrost.pipeline as bfp
import bifrost.blocks as blocks
import bifrost.views as views

from copy import deepcopy

import blocks as byip

with bfp.Pipeline() as pipeline:
    import sys
    import os
    try:
        filepath = sys.argv[1]
    except:
        print("Usage: ./gpuspec.py path_to_raw_files")
        exit()

    ############
    ## CONFIG
    ############
    
    n_blocks = 128              # Number of blocks in raw file
    n_coarse = 64
    n_t_per_coarse = 524288
    n_pol    = 4                
    n_int    = 64               # Runtime - Number of integrations
    n_chunks = 2                # Runtime - Number of time chunks 
    n_t      = n_blocks / n_int / n_chunks
    n_chan   = n_coarse * n_t_per_coarse * n_chunks
    
    # Config for mid-resolution FFT
    n_fft_midres  = 1024
    n_int_midres  = 1024
    n_gulp_midres = 8
    
    # Config for low-resolution FFT
    n_fft_lowres  = 8
    n_int_lowres  = 128
    n_gulp_lowres = 8192
    
    # Compute some stuff for HDF5 writers
    n_chan_lowres = n_fft_lowres * n_coarse
    n_chan_midres = n_fft_midres * n_coarse
    n_t_midres    = n_blocks * (n_t_per_coarse / n_fft_midres) / n_int_midres
    n_t_lowres    = n_blocks * (n_t_per_coarse / n_fft_lowres) / n_int_midres
    
    
    
    filelist = sorted(glob.glob(os.path.join(filepath, '*.raw')))
    
    # Read from guppi raw file
    b_read   = blocks.read_guppi_raw(filelist, core=0, buffer_nframe=2)
    b_read   = views.rename_axis(b_read, 'freq', 'channel')
    
    #############
    ##  HIRES  ##
    #############
    
    # Buffer up two blocks & reshape to allow longer FFT
    with bf.block_scope(fuse=True, core=1):
        b_gup2    = views.split_axis(b_read, axis='time', n=n_chunks, label='time_chunk')
        b_gup2    = blocks.transpose(b_gup2, axes=['time', 'channel', 'time_chunk', 'fine_time', 'pol'])
        b_gup2    = views.reverse_scale(b_gup2, 'time_chunk')
        b_gup2    = views.merge_axes(b_gup2, 'time_chunk', 'fine_time', label='fine_time')
        blocks.print_header(b_gup2)
   
    # Copy over to GPU and FFT
    with bf.block_scope(fuse=True, core=2):
        b_copy    = blocks.copy(b_gup2,  space='cuda')
        b_fft     = blocks.fft(b_copy,  axes='fine_time', axis_labels='freq')
        b_ffs     = blocks.fftshift(b_fft, axes='freq')
        #blocks.print_header(b_fft)
        b_pow     = blocks.detect(b_ffs, mode='stokes')
        b_copp    = blocks.copy(b_pow, space='system')
    
    #blocks.print_header(b_copp)
    # Flatten channel/freq axis to form output spectra and accumulate
    b_copp    = blocks.copy(b_copp, space='system', buffer_nframe=4, core=2)
    b_flat    = views.merge_axes(b_copp, 'channel', 'freq', label='freq')
    b_acc     = byip.accumulate(b_flat, n_int, core=2, buffer_nframe=4)
    #blocks.print_header(b_acc)
    b_hdf     = byip.hdf_writer(b_acc, shape=[n_t, n_chan, n_pol], dtype='float32', core=2)
    
    
    ##############
    ##  MIDRES  ##
    ##############
    
    # Reshape array for midres FFT
    with bf.block_scope(fuse=True, core=3):
        b_midres    = views.split_axis(b_read, axis='fine_time', n=n_fft_midres, label='fft_window') 
        b_midres    = blocks.transpose(b_midres, axes=['time', 'fine_time', 'channel', 'fft_window', 'pol'])
        b_midres    = views.reverse_scale(b_midres, 'fine_time')   
        b_midres    = views.merge_axes(b_midres, 'time', 'fine_time')
        b_midres    = blocks.copy(b_midres, space='system', gulp_nframe=n_gulp_midres)
        #blocks.print_header(b_midres)
        
    # Do midres FFT + Detect on GPU
    with bf.block_scope(fuse=True, core=4):
        b_midres    = blocks.copy(b_midres,  space='cuda')
        b_midres    = blocks.FftBlock(b_midres, axes='fft_window', axis_labels='freq')
        b_midres    = blocks.fftshift(b_midres, axes='freq')
        b_midres    = blocks.detect(b_midres, mode='stokes')
        b_midres    = blocks.accumulate(b_midres, n_int_midres)
        b_midres    = blocks.copy(b_midres, space='system')
    
    blocks.print_header(b_midres)
    b_midres     = views.merge_axes(b_midres, 'channel', 'freq')
    b_midres     = byip.hdf_writer(b_midres, shape=[n_t_midres, n_chan_midres, n_pol], 
                                    file_ext='midres.h5', dtype='float32', core=4)
    
    ##############
    ##  LOWRES  ##
    ##############
    
    # Reshape array for lowres FFT
    with bf.block_scope(fuse=True, core=5):
        b_lowres    = views.split_axis(b_read, axis='fine_time', n=n_fft_lowres, label='fft_window') 
        b_lowres    = blocks.transpose(b_lowres, axes=['time', 'fine_time', 'channel', 'fft_window', 'pol'])
        b_lowres    = views.reverse_scale(b_lowres, 'fine_time')     
        b_lowres    = views.merge_axes(b_lowres, 'time', 'fine_time')
        b_lowres    = blocks.copy(b_lowres, space='system', gulp_nframe=n_gulp_lowres)
        blocks.print_header(b_lowres)
        
    # Do lowres FFT + Detect on GPU
    with bf.block_scope(fuse=True, core=6):
        b_lowres    = blocks.copy(b_lowres,  space='cuda')
        b_lowres    = blocks.FftBlock(b_lowres, axes='fft_window', axis_labels='freq')
        b_lowres    = blocks.fftshift(b_lowres, axes='freq')
        b_lowres    = blocks.detect(b_lowres, mode='stokes')
        b_lowres    = blocks.accumulate(b_lowres, n_int_lowres)
        b_lowres    = blocks.copy(b_lowres, space='system')

    blocks.print_header(b_lowres)
    b_lowres     = views.merge_axes(b_lowres, 'channel', 'freq')
    b_lowres     = byip.hdf_writer(b_lowres, shape=[n_t_lowres, n_chan_lowres, n_pol], 
                                    file_ext='lowres.h5', dtype='float32', core=6)
    
    
    pipeline.run()
