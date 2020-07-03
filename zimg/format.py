import numpy as np
from zimg import zimg

__all__ = ['FormatCvt', 'convertFormat']

depth_map = {np.dtype('uint8'): 8, np.dtype('uint16'): 16, np.dtype('float32'): 32}

def createFormat(width, height, depth, color=None, range=None, matrix=None, transfer=None, primaries=None):
    # parameters
    if color is None:
        color = 'RGB'
    if range is None:
        range = 'LIMITED' if color.upper() == 'YUV' else 'FULL'
    if matrix is None:
        matrix = 'RGB' if color.upper() == 'RGB' else 'BT709'
    if transfer is None:
        transfer = 'BT709'
    if primaries is None:
        primaries = 'BT709'
    # create format
    f = zimg.ZFormat()
    f.width = width
    f.height = height
    f.pixel_type = zimg.Pixel.BYTE if depth <= 8 else zimg.Pixel.WORD if depth <= 16 else zimg.Pixel.FLOAT
    f.color_family = getattr(zimg.Color, color.upper())
    f.matrix_coefficients = getattr(zimg.Matrix, matrix.upper())
    f.transfer_characteristics = getattr(zimg.Transfer, transfer.upper())
    f.color_primaries = getattr(zimg.Primaries, primaries.upper())
    f.depth = depth
    f.pixel_range = getattr(zimg.Range, range.upper())
    # return format
    return f

class FormatCvt:
    def __init__(self, sw, sh, depth_in, dw=None, dh=None,
        filter=None, filter_a=None, filter_b=None, dither=None,
        color_in=None, range_in=None, matrix_in=None, transfer_in=None, primaries_in=None,
        depth=None, color=None, range=None, matrix=None, transfer=None, primaries=None):
        # basic parameters
        self.sw = sw
        self.sh = sh
        self.depth_in = depth_in
        if dw is None:
            dw = self.sw
        if dh is None:
            dh = self.sh
        if color_in is None:
            color_in = 'RGB' if matrix_in is None else 'RGB' if matrix_in.upper() in ('RGB', 'UNSPECIFIED') else 'YUV'
        if depth is None:
            depth = depth_in
        if color is None:
            color = color_in if matrix is None else 'RGB' if matrix.upper() in ('RGB', 'UNSPECIFIED') else 'YUV'
        if range is None:
            range = range_in
        if matrix is None:
            matrix = matrix_in
        if transfer is None:
            transfer = transfer_in
        if primaries is None:
            primaries = primaries_in
        # create zimg params
        params = zimg.ZGraphParams()
        if filter is not None:
            params.resample_filter = params.resample_filter_uv = getattr(zimg.Resample, filter.upper())
        if filter_a is not None:
            params.filter_param_a = params.filter_param_a_uv = filter_a
        if filter_b is not None:
            params.filter_param_b = params.filter_param_b_uv = filter_b
        if dither is not None:
            params.dither_type = getattr(zimg.Dither, dither.upper())
        # create input format
        src_format = createFormat(sw, sh, depth_in, color_in, range_in, matrix_in, transfer_in, primaries_in)
        # create output format
        dst_format = createFormat(dw, dh, depth, color, range, matrix, transfer, primaries)
        # create zimg filter
        self.zfilter = zimg.ZFilter(src_format, dst_format, params)

    def __call__(self, src, channel_first=False):
        # check input format
        depth_in = depth_map.get(src.dtype)
        rank = len(src.shape)
        if rank == 2:
            channel_first = True
        elif rank != 3:
            raise ValueError('the rank ({}) of the input should be either 2 or 3.'.format(rank))
        sw = src.shape[-1 if channel_first else -2]
        sh = src.shape[-2 if channel_first else -3]
        if depth_in != self.depth_in:
            raise ValueError('input depth {} not match the desired {}'.format(depth_in, self.depth_in))
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
    def create(cls, src, *args, channel_first=False, **kwargs):
        # parameters
        depth_in = depth_map.get(src.dtype)
        if depth_in is None:
            raise ValueError('Unsupported data type {}, must be uint8, uint16 or float32'.format(src.dtype))
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
        # decide color family if not provided
        if 'color_in' not in kwargs:
            kwargs['color_in'] = 'GREY' if channels == 1 else None
        # return ZimgFilter instance
        return cls(sw, sh, depth_in, *args, **kwargs)

def convertFormat(src, *args, channel_first=False, **kwargs):
    converter = FormatCvt.create(src, *args, channel_first=channel_first, **kwargs)
    return converter(src, channel_first=channel_first)
