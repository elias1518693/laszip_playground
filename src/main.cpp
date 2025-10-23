
#include <string>
#include <vector>
#include <print>
#include <format>

#include "laszip/laszip_api.h"

using namespace std;

struct Point{
	double x, y, z;
	uint16_t r, g, b;
};

int main(){

	string file = "E:/resources/pointclouds/bunny_small.laz";

	laszip_POINTER laszip_reader = nullptr;
	laszip_header* lazHeader = nullptr;
	laszip_point* laz_point = nullptr;

	laszip_BOOL is_compressed;
	laszip_BOOL request_reader = true;

	laszip_create(&laszip_reader);
	laszip_request_compatibility_mode(laszip_reader, request_reader);
	laszip_open_reader(laszip_reader, file.c_str(), &is_compressed);

	laszip_get_header_pointer(laszip_reader, &lazHeader);
	laszip_get_point_pointer(laszip_reader, &laz_point);
	// laszip_seek_point(laszip_reader, firstPoint);

	vector<Point> points;
	for (int i = 0; i < 10; i++) {
		double XYZ[3];
		laszip_read_point(laszip_reader);
		laszip_get_coordinates(laszip_reader, XYZ);

		Point point{};
		point.x = XYZ[0];
		point.y = XYZ[1];
		point.z = XYZ[2];

		auto rgb = laz_point->rgb;
		point.r = rgb[0] > 255 ? rgb[0] / 256 : rgb[0];
		point.g = rgb[1] > 255 ? rgb[1] / 256 : rgb[1];
		point.b = rgb[2] > 255 ? rgb[2] / 256 : rgb[2];

		points.push_back(point);
	}
	laszip_close_reader(laszip_reader);

	println("Point data: ");
	for(Point point : points){
		println("{:6.3f}, {:6.3f}, {:6.3f} | {:3d}, {:3d}, {:3d}", point.x, point.y, point.z, point.r, point.g, point.b);
	}

	return 0;
}