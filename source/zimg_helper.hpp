#pragma once

#include "zimg++.hpp"
#include <array>
#include <memory>
#include <cstddef>
#ifdef _WIN32
#include <malloc.h>
#endif

const size_t ALIGNMENT = 32;
const int MAX_PLANES = 3;

template<typename T = void>
static inline T* aligned_malloc(size_t size, size_t alignment = ALIGNMENT)
{
#ifdef _WIN32
	return reinterpret_cast<T*>(_aligned_malloc(size, alignment));
#else
	void *tmp = nullptr;
	if (posix_memalign(&tmp, alignment, size)) tmp = 0;
	return reinterpret_cast<T*>(tmp);
#endif
}

static inline void aligned_free(void *ptr)
{
#ifdef _WIN32
	_aligned_free(ptr);
#else
	free(ptr);
#endif
}

// bit-copy from one 2D array to another
// note that row_size is not width, but width * sizeof(T), based on BYTES
static inline void bitblt(void *dstp, std::ptrdiff_t dst_stride,
	const void *srcp, std::ptrdiff_t src_stride, size_t row_size, size_t height)
{
	if (height)
	{
		if (src_stride == dst_stride && src_stride == static_cast<std::ptrdiff_t>(row_size))
		{
			memcpy(dstp, srcp, row_size * height);
		}
		else
		{
			const uint8_t *srcp8 = static_cast<const uint8_t *>(srcp);
			uint8_t *dstp8 = static_cast<uint8_t *>(dstp);
			for (size_t i = 0; i < height; ++i)
			{
				memcpy(dstp8, srcp8, row_size);
				srcp8 += src_stride;
				dstp8 += dst_stride;
			}
		}
	}
}

// Deleter for aligned memory
struct AlignedDeleter
{
	void operator()(void *ptr)
	{
		aligned_free(ptr);
	}
};

// template class to store an image plane
template<typename T>
class ImagePlane
{
public:
	typedef ImagePlane<T> Tthis;
	typedef std::ptrdiff_t Tdiff;
	typedef std::shared_ptr<T> Tptr;
	typedef T *pointer;
	typedef const T *const_pointer;

	// default constructor
	ImagePlane()
		: width(0), height(0), stride(0), data(nullptr)
	{}

	// create an instance based on width and height of the image
	// stride is automatically calculated, memory allocation is implicitly performed
	// *** memory alignment is guaranteed ***
	ImagePlane(int64_t width, int64_t height)
		: width(width), height(height), stride(this->cal_stride(width)),
		data(this->allocate(this->cal_stride(width) * height))
	{}

	// create an instance refering to the existing data
	// stride is inherited, memory allocation is handled by the user
	// *** no guarantee about the memory alignment ***
	ImagePlane(int64_t width, int64_t height, Tdiff stride, void *data)
		: width(width), height(height), stride(stride),
		data(static_cast<pointer>(data), [](void *ptr) {})
	{}

	// create an instance refering to the existing data
	// stride is inherited
	// shared_ptr is used to automatically handle memory deallocation
	// *** no guarantee about the memory alignment ***
	ImagePlane(int64_t width, int64_t height, Tdiff stride, const Tptr &data)
		: width(width), height(height), stride(stride),
		data(data)
	{}

	// copy data from an existing one
	// both should have the same width and height
	Tthis &from(Tdiff stride, const void *data)
	{
		bitblt(this->data.get(), this->stride, data, stride,
			this->width * sizeof(T), this->height);
		return *this;
	}

	// copy data to an external one
	// both should have the same width and height
	const Tthis &to(Tdiff stride, void *data) const
	{
		bitblt(data, stride, this->data.get(), this->stride,
			this->width * sizeof(T), this->height);
		return *this;
	}

	// return a deep copy of the current instance
	// the copy has the same stride
	Tthis copy() const
	{
		Tptr new_data = this->allocate(this->stride * this->height);
		bitblt(new_data.get(), this->stride, this->data.get(), this->stride,
			this->width * sizeof(T), this->height);
		return Tthis(this->width, this->height, this->stride, new_data);
	}

	int64_t getWidth() const { return this->width; }
	int64_t getHeight() const { return this->height; }
	Tdiff getStride() const { return this->stride; }
	const Tptr &getPtr() const { return this->data; }
	const_pointer getData() const { return this->data.get(); }
	pointer getData() { return this->data.get(); }

	// Judge if the data and stride matches the specified memory alignment
	bool isAligned(size_t alignment = ALIGNMENT) const
	{
		return (reinterpret_cast<size_t>(this->data.get()) % alignment == 0)
			&& (this->stride % alignment == 0);
	}

	// static function to calculate minimum stride match the required memory alignment
	static Tdiff cal_stride(int64_t width, size_t alignment = ALIGNMENT)
	{
		return (static_cast<size_t>(width) * sizeof(T) + alignment - 1) / alignment * alignment;
	}

	// static function to allocate aligned memory and return a smart pointer
	static Tptr allocate(size_t size, size_t alignment = ALIGNMENT)
	{
		return Tptr(aligned_malloc<T>(size, alignment), AlignedDeleter());
	}

protected:
	int64_t width;
	int64_t height;
	Tdiff stride;
	Tptr data;
};

// template class to store an image with one or multiple planes
template<typename T>
class Image
{
public:
	typedef Image<T> Tthis;
	typedef ImagePlane<T> Tplane;
	typedef std::array<Tplane, MAX_PLANES> TplaneArr;
	typedef std::ptrdiff_t Tdiff;
	typedef std::shared_ptr<T> Tptr;
	typedef T *pointer;
	typedef const T *const_pointer;

	// default constructor
	Image()
		: num_planes(0)
	{}

	// create an image with a single plane
	explicit Image(const Tplane &plane0)
		: num_planes(1)
	{
		this->planes[0] = plane0;
	}

	// create an image with three planes
	Image(const Tplane &plane0, const Tplane &plane1, const Tplane &plane2)
		: num_planes(3)
	{
		this->planes[0] = plane0;
		this->planes[1] = plane1;
		this->planes[2] = plane2;
	}

	int getNumPlanes() const { return this->num_planes; }
	const Tplane &getPlane(int p = 0) const { return this->planes[p]; }
	int64_t getWidth(int p = 0) const { return this->planes[p].getWidth(); }
	int64_t getHeight(int p = 0) const { return this->planes[p].getHeight(); }
	Tdiff getStride(int p = 0) const { return this->planes[p].getStride(); }
	const Tptr &getPtr(int p = 0) const { return this->planes[p].getPtr(); }
	const_pointer getData(int p = 0) const { return this->planes[p].getData(); }
	pointer getData(int p = 0) { return this->planes[p].getData(); }

protected:
	int num_planes;
	TplaneArr planes;
};

struct ZResizeParams
{
	// format parameters
	zimg_pixel_type_e pixel_type = ZIMG_PIXEL_BYTE;
	zimg_color_family_e color_family = ZIMG_COLOR_GREY;
	unsigned depth = 8;
	zimg_pixel_range_e pixel_range = ZIMG_RANGE_FULL;
	// graph parameters
	zimg_resample_filter_e filter = ZIMG_RESIZE_BICUBIC;
	double filter_a = NAN;
	double filter_b = NAN;
	zimg_dither_type_e dither_type = ZIMG_DITHER_NONE;
	zimg_cpu_type_e cpu_type = ZIMG_CPU_AUTO;

	static ZResizeParams build(int planes = 1, unsigned depth = 8)
	{
		ZResizeParams params;
		params.pixel_type = depth > 16 ? ZIMG_PIXEL_FLOAT : depth > 8 ? ZIMG_PIXEL_WORD : ZIMG_PIXEL_BYTE;
		params.color_family = planes > 1 ? ZIMG_COLOR_RGB : ZIMG_COLOR_GREY;
		params.depth = depth;
		return params;
	}
};

class ZFilter
{
public:
	typedef ZFilter Tthis;
	typedef zimgxx::zimage_format Zformat;
	typedef zimgxx::zfilter_graph_builder_params Zparams;
	typedef zimgxx::FilterGraph Zgraph;
	typedef zimgxx::zimage_buffer Zbuffer;
	typedef zimgxx::zimage_buffer_const ZbufferC;
	typedef std::unique_ptr<void, AlignedDeleter> TempPtr;

	// create an instance based on zimage format and zfilter graph params
	ZFilter(const Zformat &src_format, const Zformat &dst_format, const Zparams &params)
	{
		// build graph
		this->init(src_format, dst_format, params);
	}

	// create an instance based on a custom resize parameters
	// can only perform resizing without other colorspace conversions
	ZFilter(const ZResizeParams &params,
		unsigned src_width, unsigned src_height, unsigned dst_width, unsigned dst_height,
		double roi_left = 0, double roi_top = 0, double roi_width = 0, double roi_height = 0)
	{
		// source format
		Zformat src_format;
		src_format.width = src_width;
		src_format.height = src_height;
		src_format.pixel_type = params.pixel_type;
		src_format.color_family = params.color_family;
		src_format.depth = params.depth;
		src_format.pixel_range = params.pixel_range;
		src_format.active_region.left = roi_left;
		src_format.active_region.top = roi_top;
		src_format.active_region.width = roi_width > 0 ? roi_width : src_width - roi_width;
		src_format.active_region.height = roi_height > 0 ? roi_height : src_height - roi_height;
		// target format
		Zformat dst_format;
		dst_format.width = dst_width;
		dst_format.height = dst_height;
		dst_format.pixel_type = params.pixel_type;
		dst_format.color_family = params.color_family;
		dst_format.depth = params.depth;
		dst_format.pixel_range = params.pixel_range;
		// graph parameters
		Zparams g_params;
		g_params.resample_filter = params.filter;
		g_params.filter_param_a = params.filter_a;
		g_params.filter_param_b = params.filter_b;
		g_params.resample_filter_uv = params.filter;
		g_params.filter_param_a_uv = params.filter_a;
		g_params.filter_param_b_uv = params.filter_b;
		g_params.dither_type = params.dither_type;
		g_params.cpu_type = params.cpu_type;
		// build graph
		this->init(src_format, dst_format, g_params);
	}

	// perform conversion on image data pointer (=ZIMG_COLOR_GREY)
	void operator()(void *dst, const void *src,
		std::ptrdiff_t dst_stride, std::ptrdiff_t src_stride)
	{
		Zbuffer buf_dst;
		ZbufferC buf_src;
		buf_src.data(0) = src;
		buf_src.stride(0) = src_stride;
		buf_src.mask(0) = ZIMG_BUFFER_MAX;
		buf_dst.data(0) = dst;
		buf_dst.stride(0) = dst_stride;
		buf_dst.mask(0) = ZIMG_BUFFER_MAX;
		this->graph.process(buf_src, buf_dst, this->tmp_buf.get());
	}

	// perform conversion on image data pointer (!=ZIMG_COLOR_GREY)
	void operator()(std::array<void *, MAX_PLANES> dst, std::array<const void *, MAX_PLANES> src,
		std::array<std::ptrdiff_t, MAX_PLANES> dst_stride, std::array<std::ptrdiff_t, MAX_PLANES> src_stride)
	{
		Zbuffer buf_dst;
		ZbufferC buf_src;
		for (int p = 0; p < MAX_PLANES; ++p)
		{
			buf_src.data(p) = src[p];
			buf_src.stride(p) = src_stride[p];
			buf_src.mask(p) = ZIMG_BUFFER_MAX;
		}
		for (int p = 0; p < MAX_PLANES; ++p)
		{
			buf_dst.data(p) = dst[p];
			buf_dst.stride(p) = dst_stride[p];
			buf_dst.mask(p) = ZIMG_BUFFER_MAX;
		}
		this->graph.process(buf_src, buf_dst, this->tmp_buf.get());
	}

	// perform conversion on ImagePlane
	// should be called only when color family is ZIMG_COLOR_GREY
	template<typename T>
	void operator()(ImagePlane<T> &dst, const ImagePlane<T> &src)
	{
		this->operator()(dst.getData(), src.getData(),
			dst.getStride(), src.getStride());
	}

	// perform conversion on Image
	// should not be called when color family is ZIMG_COLOR_GREY
	template<typename T>
	void operator()(Image<T> &dst, const Image<T> &src)
	{
		Zbuffer buf_dst;
		ZbufferC buf_src;
		for (int p = 0; p < src.getNumPlanes(); ++p)
		{
			buf_src.data(p) = src.getData(p);
			buf_src.stride(p) = src.getStride(p);
			buf_src.mask(p) = ZIMG_BUFFER_MAX;
		}
		for (int p = 0; p < dst.getNumPlanes(); ++p)
		{
			buf_dst.data(p) = dst.getData(p);
			buf_dst.stride(p) = dst.getStride(p);
			buf_dst.mask(p) = ZIMG_BUFFER_MAX;
		}
		this->graph.process(buf_src, buf_dst, this->tmp_buf.get());
	}

protected:
	Zformat src_format;
	Zformat dst_format;
	Zparams params;
	Zgraph graph;
	TempPtr tmp_buf;

	// disable copy constructor and copy assignment, as tmp_buf should not be shared
	ZFilter(const Tthis &other) = delete;
	Tthis &operator=(const Tthis &other) = delete;

	// initialize zimage parameters, zimage filter graph and temporary buffer
	void init(const Zformat &src_format, const Zformat &dst_format, const Zparams &params)
	{
		this->src_format = src_format;
		this->dst_format = dst_format;
		this->params = params;
		this->graph = Zgraph::build(src_format, dst_format, &params);
		this->tmp_buf = TempPtr(aligned_malloc(this->graph.get_tmp_size(), ALIGNMENT), AlignedDeleter());
	}
};
