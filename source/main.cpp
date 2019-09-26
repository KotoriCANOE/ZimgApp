#include "zimg_helper.hpp"
#include <iostream>

typedef float DTYPE;
const unsigned SW = 16;
const unsigned DW = 8;
const unsigned SH = 1;
const unsigned DH = 1;
const std::array<DTYPE, SW> src_row = { 0, 1, 2, 3, 4, 5, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5 };

int main(int argc, char **argv)
{
	// create Image
	ImagePlane<DTYPE> src(SW, SH), dst(DW, DH);
	src.from(sizeof(src_row), src_row.data());
	// zimg API
	std::cout << "API version: " << zimg_get_api_version(nullptr, nullptr) << std::endl;
	// create ZFilter instance
	ZResizeParams params = ZResizeParams::build(1, 32);
	ZFilter zfilter(params, SW, SH, DW, DH);
	// apply resizing
	zfilter(dst, src);
	// print result
	auto src_data = src.getData();
	auto dst_data = dst.getData();
	std::copy(src_data, src_data + SW, std::ostream_iterator<DTYPE>(std::cout, ","));
	std::cout << std::endl;
	std::copy(dst_data, dst_data + DW, std::ostream_iterator<DTYPE>(std::cout, ","));
	std::cout << std::endl;
	// exit
	std::getchar();
	return 0;
}
