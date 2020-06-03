import os
import sys
import numpy as np
from zimg import zimg

__all__ = ['ZimgFilter', 'zscale', 'zresize']

class ZimgFilter:
    depth_map = {np.dtype('uint8'): 8, np.dtype('uint16'): 16, np.dtype('float32'): 32}
    filter_map = {'bicubic': zimg.Resample.BICUBIC, 'bilinear': zimg.Resample.BILINEAR,
        'lanczos': zimg.Resample.LANCZOS, 'point': zimg.Resample.POINT,
        'spline16': zimg.Resample.SPLINE16, 'spline36': zimg.Resample.SPLINE36, 'spline64': zimg.Resample.SPLINE64}
    
    def __init__(self, depth, channels, sw, sh, dw, dh, filter=None, filter_a=None, filter_b=None):
        self.depth = depth
        self.channels = channels
        self.sw = sw
        self.sh = sh
        # create zimg params
        params = zimg.ZResizeParams.build(channels, depth)
        if filter is not None:
            filter_id = self.filter_map.get(filter.lower())
            if filter_id is None:
                raise ValueError('Unsupported filter: {}'.format(filter))
            params.filter = filter_id
        if filter_a is not None:
            params.filter_a = filter_a
        if filter_b is not None:
            params.filter_b = filter_b
        # create zimg filter
        self.zfilter = zimg.ZFilter(params, sw, sh, dw, dh)
    
    def __call__(self, src, channel_first=False):
        # check input format
        depth = self.depth_map.get(src.dtype)
        rank = len(src.shape)
        if rank == 2:
            channels = 1
        elif rank == 3:
            channels = src.shape[-3 if channel_first else -1]
        else:
            raise ValueError('the rank ({}) of the input should be either 2 or 3.'.format(rank))
        sw = src.shape[-1 if channel_first else -2]
        sh = src.shape[-2 if channel_first else -3]
        if depth != self.depth:
            raise ValueError('input depth {} not match the desired {}'.format(depth, self.depth))
        if channels != self.channels:
            raise ValueError('input channels {} not match the desired {}'.format(channels, self.channels))
        if sw != self.sw or sh != self.sh:
            raise ValueError('input size {}x{} not match the desired {}x{}'.format(sw, self.sw, sh, self.sh))
        # apply filter
        if not channel_first:
            src = np.transpose(src, (2, 0, 1))
        dst = self.zfilter(src)
        if not channel_first:
            dst = np.transpose(dst, (1, 2, 0))
        # return
        return dst

    @classmethod
    def create(cls, src, dw, dh, filter=None, filter_a=None, filter_b=None, channel_first=False):
        # parameters
        depth = cls.depth_map.get(src.dtype)
        if depth is None:
            raise ValueError('Unsupported data type {}, must be uint8, uint16 or float32'.format(src.dtype))
        channels = src.shape[-3 if channel_first else -1]
        sw = src.shape[-1 if channel_first else -2]
        sh = src.shape[-2 if channel_first else -3]
        # return ZimgFilter instance
        return cls(depth, channels, sw, sh, dw, dh, filter, filter_a, filter_b)
    
    @classmethod
    def createScale(cls, src, scale, filter=None, filter_a=None, filter_b=None, channel_first=False):
        sw = src.shape[-1 if channel_first else -2]
        sh = src.shape[-2 if channel_first else -3]
        dw = int(sw * scale + 0.5)
        dh = int(sh * scale + 0.5)
        return cls.create(src, dw, dh, filter, filter_a, filter_b)

def zscale(src, scale, filter=None, filter_a=None, filter_b=None, channel_first=False):
    zfilter = ZimgFilter.createScale(src, scale, filter, filter_a, filter_b, channel_first)
    return zfilter(src, channel_first)

def zresize(src, dw, dh, filter=None, filter_a=None, filter_b=None, channel_first=False):
    zfilter = ZimgFilter.create(src, dw, dh, filter, filter_a, filter_b, channel_first)
    return zfilter(src, channel_first)
