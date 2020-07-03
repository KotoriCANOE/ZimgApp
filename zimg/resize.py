import numpy as np
from zimg import zimg

__all__ = ['Resizer', 'scale', 'resize']

depth_map = {np.dtype('uint8'): 8, np.dtype('uint16'): 16, np.dtype('float32'): 32}

class Resizer:
    def __init__(self, depth, channels, sw, sh, dw, dh,
        filter=None, filter_a=None, filter_b=None, dither=None,
        roi_left=0, roi_top=0, roi_width=0, roi_height=0):
        self.depth = depth
        self.channels = channels
        self.sw = sw
        self.sh = sh
        # create zimg params
        params = zimg.ZResizeParams.build(channels, depth)
        if filter is not None:
            params.filter = getattr(zimg.Resample, filter.upper())
        if filter_a is not None:
            params.filter_a = filter_a
        if filter_b is not None:
            params.filter_b = filter_b
        if dither is not None:
            params.dither_type = getattr(zimg.Dither, dither.upper())
        # create zimg filter
        self.zfilter = zimg.ZFilter(params, sw, sh, dw, dh,
            roi_left, roi_top, roi_width, roi_height)
    
    def __call__(self, src, channel_first=False):
        # check input format
        depth = depth_map.get(src.dtype)
        rank = len(src.shape)
        if rank == 2:
            channels = 1
            channel_first = True
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
            raise ValueError('input size {}x{} not match the desired {}x{}'.format(sw, sh, self.sw, self.sh))
        # apply filter
        if rank == 3 and not channel_first:
            src = np.transpose(src, (2, 0, 1))
        dst = self.zfilter(src)
        if rank == 3 and not channel_first:
            dst = np.transpose(dst, (1, 2, 0))
        # return
        return dst

    @classmethod
    def create(cls, src, dw, dh, *args, channel_first=False, **kwargs):
        # parameters
        depth = depth_map.get(src.dtype)
        if depth is None:
            raise ValueError('Unsupported data type {}, must be uint8, uint16 or float32'.format(src.dtype))
        rank = len(src.shape)
        if rank == 2:
            channels = 1
            channel_first = True
        elif rank == 3:
            channels = src.shape[-3 if channel_first else -1]
        sw = src.shape[-1 if channel_first else -2]
        sh = src.shape[-2 if channel_first else -3]
        # return ZimgFilter instance
        return cls(depth, channels, sw, sh, dw, dh, *args, **kwargs)
    
    @classmethod
    def createScale(cls, src, scale, *args, channel_first=False, **kwargs):
        rank = len(src.shape)
        if rank == 2:
            channel_first = True
        sw = src.shape[-1 if channel_first else -2]
        sh = src.shape[-2 if channel_first else -3]
        dw = int(sw * scale + 0.5)
        dh = int(sh * scale + 0.5)
        return cls.create(src, dw, dh, *args, channel_first=channel_first, **kwargs)

def scale(src, scale, *args, channel_first=False, **kwargs):
    resizer = Resizer.createScale(src, scale, *args, channel_first=channel_first, **kwargs)
    return resizer(src, channel_first=channel_first)

def resize(src, dw, dh, *args, channel_first=False, **kwargs):
    resizer = Resizer.create(src, dw, dh, *args, channel_first=channel_first, **kwargs)
    return resizer(src, channel_first=channel_first)
