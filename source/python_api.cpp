#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "zimg_helper.hpp"

////////

namespace py = pybind11;

template <typename T>
using PyArr = py::array_t<T, py::array::c_style | py::array::forcecast>;

////////

class ZFilterPy
	: public ZFilter
{
public:
	typedef ZFilterPy Tthis;
	typedef ZFilter Tbase;

	// create an instance based on zimage format and zfilter graph params
	ZFilterPy(const Zformat &src_format, const Zformat &dst_format, const Zparams &params)
		: Tbase(src_format, dst_format, params)
	{}

	// create an instance based on a custom resize parameters
	// can only perform resizing without other colorspace conversions
	ZFilterPy(const ZResizeParams &params,
		unsigned src_width, unsigned src_height, unsigned dst_width, unsigned dst_height,
		double roi_left = 0, double roi_top = 0, double roi_width = 0, double roi_height = 0)
		: Tbase(params, src_width, src_height, dst_width, dst_height,
			roi_left, roi_top, roi_width, roi_height)
	{}

	template <typename T>
	PyArr<T> __call__(PyArr<T> src_arr)
	{
		// src protobuf
		py::buffer_info src_buf = src_arr.request();
		const std::vector<ssize_t> src_shape = src_buf.shape;
		const ssize_t ndim = src_buf.ndim;
		const ssize_t src_width = src_shape[ndim - 1];
		const ssize_t src_height = src_shape[ndim - 2];
		const ssize_t channels = ndim > 2 ? src_shape[ndim - 3] : 1;

		if (ndim < 2 || ndim > 3)
		{
			throw std::runtime_error("Number of dimensions must be 2 or 3");
		}
		if (channels != 1 && channels != 3)
		{
			throw std::runtime_error("Number of channels must be 1 or 3 (CHW format)");
		}
		if (src_width != this->src_format.width || src_height != this->src_format.height)
		{
			throw std::runtime_error("Input width and height must match the format defined in the filter");
		}

		// allocate temp memory
		const ssize_t dst_width = this->dst_format.width;
		const ssize_t dst_height = this->dst_format.height;
		ImagePlane<T> src_image(src_width, src_height * channels); // allocate CHW data
		ImagePlane<T> dst_image(dst_width, dst_height * channels); // allocate CHW data

		// dst protobuf
		std::vector<ssize_t> dst_shape = src_buf.shape;
		dst_shape[ndim - 1] = dst_width;
		dst_shape[ndim - 2] = dst_height;
		PyArr<T> dst_arr(dst_shape);
		py::buffer_info dst_buf = dst_arr.request();

		// copy src data to aligned memory
		const std::vector<ssize_t> src_strides = src_buf.strides;
		const ssize_t src_stride_c = ndim > 2 ? src_strides[ndim - 3] : 0;
		const ssize_t src_stride_h = src_strides[ndim - 2];
		const ssize_t src_stride_w = src_strides[ndim - 1];

		if (src_stride_w != sizeof(T)) // not continuous elements, element-wise copy
		{
			for (ssize_t c = 0; c < channels; ++c)
			{
				for (ssize_t h = 0; h < src_height; ++h)
				{
					const uint8_t *origin_ptr = static_cast<const uint8_t *>(src_buf.ptr)
						+ c * src_stride_c + h * src_stride_h;
					uint8_t *target_ptr = reinterpret_cast<uint8_t *>(src_image.getData())
						+ (c * src_height + h) * src_image.getStride();

					for (const uint8_t *origin_upper = origin_ptr + src_width * src_stride_w; origin_ptr < origin_upper;
						origin_ptr += src_stride_w, target_ptr += sizeof(T))
					{
						*reinterpret_cast<T *>(target_ptr) = *reinterpret_cast<const T *>(origin_ptr);
					}
				}
			}
		}
		else // continuous elements, not aligned data, copy with bitblt
		{
			for (ssize_t c = 0; c < channels; ++c)
			{
				const uint8_t *origin_ptr = static_cast<const uint8_t *>(src_buf.ptr)
					+ c * src_stride_c;
				uint8_t *target_ptr = reinterpret_cast<uint8_t *>(src_image.getData())
					+ c * src_height * src_image.getStride();

				bitblt(target_ptr, src_image.getStride(), origin_ptr,
					src_stride_h, src_width * sizeof(T), src_height);
			}
		}

		// process
		if (channels == 1)
		{
			Tbase::operator()(dst_image, src_image);
		}
		else // channels == 3
		{
			const uint8_t *src_data = reinterpret_cast<const uint8_t *>(src_image.getData());
			uint8_t *dst_data = reinterpret_cast<uint8_t *>(dst_image.getData());

			const ssize_t src_stride_h = src_image.getStride();
			const ssize_t dst_stride_h = dst_image.getStride();
			const ssize_t src_stride_c = src_height * src_stride_h;
			const ssize_t dst_stride_c = dst_height * dst_stride_h;

			Tbase::operator()({ dst_data, dst_data + dst_stride_c, dst_data + 2 * dst_stride_c },
				{ src_data, src_data + src_stride_c, src_data + 2 * src_stride_c },
				{ dst_stride_h, dst_stride_h, dst_stride_h },
				{ src_stride_h, src_stride_h, src_stride_h });
		}

		// copy dst data from aligned memory
		const std::vector<ssize_t> dst_strides = dst_buf.strides;
		const ssize_t dst_stride_c = ndim > 2 ? dst_strides[ndim - 3] : 0;
		const ssize_t dst_stride_h = dst_strides[ndim - 2];
		const ssize_t dst_stride_w = dst_strides[ndim - 1];

		if (dst_stride_w != sizeof(T)) // not continuous elements, element-wise copy
		{
			for (ssize_t c = 0; c < channels; ++c)
			{
				for (ssize_t h = 0; h < dst_height; ++h)
				{
					const uint8_t *origin_ptr = reinterpret_cast<const uint8_t *>(dst_image.getData())
						+ (c * dst_height + h) * dst_image.getStride();
					uint8_t *target_ptr = static_cast<uint8_t *>(dst_buf.ptr)
						+ c * dst_stride_c + h * dst_stride_h;

					for (const uint8_t *origin_upper = origin_ptr + dst_width * sizeof(T); origin_ptr < origin_upper;
						origin_ptr += sizeof(T), target_ptr += dst_stride_w)
					{
						*reinterpret_cast<T *>(target_ptr) = *reinterpret_cast<const T *>(origin_ptr);
					}
				}
			}
		}
		else // continuous elements, not aligned data, copy with bitblt
		{
			for (ssize_t c = 0; c < channels; ++c)
			{
				const uint8_t *origin_ptr = reinterpret_cast<const uint8_t *>(dst_image.getData())
					+ c * dst_height * dst_image.getStride();
				uint8_t *target_ptr = static_cast<uint8_t *>(dst_buf.ptr)
					+ c * dst_stride_c;

				bitblt(target_ptr, dst_stride_h, origin_ptr,
					dst_image.getStride(), dst_width * sizeof(T), dst_height);
			}
		}

		// return array
		return dst_arr;
	}

protected:
	// disable copy constructor and copy assignment, as tmp_buf should not be shared
	ZFilterPy(const Tthis &other) = delete;
	Tthis &operator=(const Tthis &other) = delete;
};

////////

using namespace pybind11::literals;

PYBIND11_MODULE(zimg, m)
{
	m.doc() = "Zimg: a plugin for colorspace conversion";
	////////
	// Enumerators
	py::enum_<zimg_cpu_type_e>(m, "CPU")
		.value("NONE", ZIMG_CPU_NONE)
		.value("AUTO", ZIMG_CPU_AUTO)
		.value("AUTO_64B", ZIMG_CPU_AUTO_64B);
	py::enum_<zimg_pixel_type_e>(m, "Pixel")
		.value("BYTE", ZIMG_PIXEL_BYTE)
		.value("WORD", ZIMG_PIXEL_WORD)
		.value("HALF", ZIMG_PIXEL_HALF)
		.value("FLOAT", ZIMG_PIXEL_FLOAT);
	py::enum_<zimg_pixel_range_e>(m, "Range")
		.value("INTERNAL", ZIMG_RANGE_INTERNAL)
		.value("LIMITED", ZIMG_RANGE_LIMITED)
		.value("FULL", ZIMG_RANGE_FULL);
	py::enum_<zimg_color_family_e>(m, "Color")
		.value("GREY", ZIMG_COLOR_GREY)
		.value("RGB", ZIMG_COLOR_RGB)
		.value("YUV", ZIMG_COLOR_YUV);
	py::enum_<zimg_field_parity_e>(m, "Field")
		.value("PROGRESSIVE", ZIMG_FIELD_PROGRESSIVE)
		.value("TOP", ZIMG_FIELD_TOP)
		.value("BOTTOM", ZIMG_FIELD_BOTTOM);
	py::enum_<zimg_chroma_location_e>(m, "Chroma")
		.value("INTERNAL", ZIMG_CHROMA_INTERNAL)
		.value("LEFT", ZIMG_CHROMA_LEFT)
		.value("CENTER", ZIMG_CHROMA_CENTER)
		.value("TOP_LEFT", ZIMG_CHROMA_TOP_LEFT)
		.value("TOP", ZIMG_CHROMA_TOP)
		.value("BOTTOM_LEFT", ZIMG_CHROMA_BOTTOM_LEFT)
		.value("BOTTOM", ZIMG_CHROMA_BOTTOM);
	py::enum_<zimg_matrix_coefficients_e>(m, "Matrix")
		.value("INTERNAL", ZIMG_MATRIX_INTERNAL)
		.value("RGB", ZIMG_MATRIX_RGB)
		.value("BT709", ZIMG_MATRIX_BT709)
		.value("UNSPECIFIED", ZIMG_MATRIX_UNSPECIFIED)
		.value("FCC", ZIMG_MATRIX_FCC)
		.value("BT470_BG", ZIMG_MATRIX_BT470_BG)
		.value("ST170_M", ZIMG_MATRIX_ST170_M)
		.value("ST240_M", ZIMG_MATRIX_ST240_M)
		.value("YCGCO", ZIMG_MATRIX_YCGCO)
		.value("BT2020_NCL", ZIMG_MATRIX_BT2020_NCL)
		.value("BT2020_CL", ZIMG_MATRIX_BT2020_CL)
		.value("CHROMATICITY_DERIVED_NCL", ZIMG_MATRIX_CHROMATICITY_DERIVED_NCL)
		.value("CHROMATICITY_DERIVED_CL", ZIMG_MATRIX_CHROMATICITY_DERIVED_CL)
		.value("ICTCP", ZIMG_MATRIX_ICTCP);
	py::enum_<zimg_transfer_characteristics_e>(m, "Transfer")
		.value("INTERNAL", ZIMG_TRANSFER_INTERNAL)
		.value("BT709", ZIMG_TRANSFER_BT709)
		.value("UNSPECIFIED", ZIMG_TRANSFER_UNSPECIFIED)
		.value("BT470_M", ZIMG_TRANSFER_BT470_M)
		.value("BT470_BG", ZIMG_TRANSFER_BT470_BG)
		.value("BT601", ZIMG_TRANSFER_BT601)
		.value("ST240_M", ZIMG_TRANSFER_ST240_M)
		.value("LINEAR", ZIMG_TRANSFER_LINEAR)
		.value("LOG_100", ZIMG_TRANSFER_LOG_100)
		.value("LOG_316", ZIMG_TRANSFER_LOG_316)
		.value("IEC_61966_2_4", ZIMG_TRANSFER_IEC_61966_2_4)
		.value("IEC_61966_2_1", ZIMG_TRANSFER_IEC_61966_2_1)
		.value("BT2020_10", ZIMG_TRANSFER_BT2020_10)
		.value("BT2020_12", ZIMG_TRANSFER_BT2020_12)
		.value("ST2084", ZIMG_TRANSFER_ST2084)
		.value("ARIB_B67", ZIMG_TRANSFER_ARIB_B67);
	py::enum_<zimg_color_primaries_e>(m, "Primaries")
		.value("INTERNAL", ZIMG_PRIMARIES_INTERNAL)
		.value("BT709", ZIMG_PRIMARIES_BT709)
		.value("UNSPECIFIED", ZIMG_PRIMARIES_UNSPECIFIED)
		.value("BT470_M", ZIMG_PRIMARIES_BT470_M)
		.value("BT470_BG", ZIMG_PRIMARIES_BT470_BG)
		.value("ST170_M", ZIMG_PRIMARIES_ST170_M)
		.value("ST240_M", ZIMG_PRIMARIES_ST240_M)
		.value("FILM", ZIMG_PRIMARIES_FILM)
		.value("BT2020", ZIMG_PRIMARIES_BT2020)
		.value("ST428", ZIMG_PRIMARIES_ST428)
		.value("ST431_2", ZIMG_PRIMARIES_ST431_2)
		.value("ST432_1", ZIMG_PRIMARIES_ST432_1)
		.value("EBU3213_E", ZIMG_PRIMARIES_EBU3213_E);
	py::enum_<zimg_dither_type_e>(m, "Dither")
		.value("NONE", ZIMG_DITHER_NONE)
		.value("ORDERED", ZIMG_DITHER_ORDERED)
		.value("RANDOM", ZIMG_DITHER_RANDOM)
		.value("ERROR_DIFFUSION", ZIMG_DITHER_ERROR_DIFFUSION);
	py::enum_<zimg_resample_filter_e>(m, "Resample")
		.value("POINT", ZIMG_RESIZE_POINT)
		.value("BILINEAR", ZIMG_RESIZE_BILINEAR)
		.value("BICUBIC", ZIMG_RESIZE_BICUBIC)
		.value("SPLINE16", ZIMG_RESIZE_SPLINE16)
		.value("SPLINE36", ZIMG_RESIZE_SPLINE36)
		.value("LANCZOS", ZIMG_RESIZE_LANCZOS);
	////////
	// ZResizeParams
	py::class_<ZResizeParams> zresize_params(m, "ZResizeParams");
	zresize_params
		.def(py::init<>())
		.def_static("build", &ZResizeParams::build, "planes"_a=1, "depth"_a=8)
		// attributes
		.def_readwrite("pixel_type", &ZResizeParams::pixel_type)
		.def_readwrite("color_family", &ZResizeParams::color_family)
		.def_readwrite("depth", &ZResizeParams::depth)
		.def_readwrite("pixel_range", &ZResizeParams::pixel_range)
		.def_readwrite("filter", &ZResizeParams::filter)
		.def_readwrite("filter_a", &ZResizeParams::filter_a)
		.def_readwrite("filter_b", &ZResizeParams::filter_b)
		.def_readwrite("dither_type", &ZResizeParams::dither_type)
		.def_readwrite("cpu_type", &ZResizeParams::cpu_type)
		;
	// ZFilter
	py::class_<ZFilterPy> zfilter(m, "ZFilter");
	zfilter
		// constructors
		.def(py::init<const ZResizeParams &,
			unsigned, unsigned, unsigned, unsigned,
			double, double, double, double>(),
			"params"_a,
			"src_width"_a, "src_height"_a, "dst_width"_a, "dst_height"_a,
			"roi_left"_a=0, "roi_top"_a=0, "roi_width"_a=0, "roi_height"_a=0)
		// process
		.def("__call__", &ZFilterPy::__call__<uint8_t>, "Process uint8 array input")
		.def("__call__", &ZFilterPy::__call__<uint16_t>, "Process uint16 array input")
		.def("__call__", &ZFilterPy::__call__<float>, "Process float32 array input")
		;
}
