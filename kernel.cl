__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void calculateHistogram(read_only image2d_t src_image, __global int* histogram)
{
	// get pixel coordinates 
	int2 coordinates = (int2)(get_global_id(0), get_global_id(1));

	// count pixel
	float4 pixel = read_imagef(src_image, sampler, coordinates);

	// convert to RGB
	int x = (int)(pixel.x / 1.0f * 255);
	int y = (int)(pixel.y / 1.0f * 255);
	int z = (int)(pixel.z / 1.0f * 255);

	// get luminance value
	int Y = (int)(0.299f * x + 0.587f * y + 0.114f * z);
	
	// validate luminace value 
	Y  = min(max(0, Y), 255);

	// Increment luminance value count with atomic operation
	atomic_inc(&histogram[Y]);
}

__kernel void HistogramEqualization(read_only image2d_t src_image, write_only image2d_t dst_image, __global int* histogram)
{
	// get pixel coordinates
	int width = get_global_size(0);
	int height = get_global_size(1);
	int2 coordinates = (int2)(get_global_id(0), get_global_id(1));

	// pixel count
	float4 pixel = read_imagef(src_image, sampler, coordinates);

	// convert RGB range
	int R = (int)(pixel.x / 1.0f * 255);
	int G = (int)(pixel.y / 1.0f * 255);
	int B = (int)(pixel.z / 1.0f * 255);

	// convert to YUV
	int Y = (int)(0.299f * R + 0.587f * G + 0.114f * B);
	int U = (int)(-0.14713f * R - 0.28886f * G + 0.436f * B);
	int V = (int)(0.615f * R - 0.51499f * G - 0.10001f * B);

	// validate luminace value 
	Y  = min(max(0, Y), 255);

	// calculate Y
	float sum = 0.0f;
	for (int i = 0; i < Y; i++)
	{
		sum += histogram[i];	
	}

	float adjustedY = (255.0f / (width * height)) * sum;

	R = min(max(0, (int)(adjustedY + (1.13983 * V))), 255);
	G = min(max(0, (int)(adjustedY - (0.39465 * U) - (0.58060 * V))), 255);
	B = min(max(0, (int)(adjustedY + (2.03211 * U))), 255);

	// normalized RGB
	float nR = (float)((R / 255.0f) * 1.0f);
	float nG = (float)((G / 255.0f) * 1.0f);
	float nB = (float)((B / 255.0f) * 1.0f);

	float4 rgba = (float4)(nR, nG, nB, pixel.w);

	// write to output 
	write_imagef(dst_image, coordinates, rgba);
}

// 3x3 sharpening filter
__constant float sharpeningFilter[9] = { 0.0,	-1.0 / 6,		0.0,
											-1.0 / 6,	10.0 / 6,		-1.0 / 6,
											0.0,	-1.0 / 6,		0.0 };

__kernel void increaseSharpness(read_only image2d_t src_image, write_only image2d_t dst_image) {
	/* Get work-item’s row and column position */
	int column = get_global_id(0);
	int row = get_global_id(1);

	/* Accumulated pixel value */
	float4 sum = (float4)(0.0);

	/* Filter's current index */
	int filter_index = 0;

	int2 coord;
	float4 pixel;

	/* Iterate over the rows */
	for (int i = -1; i <= 1; i++) {
		coord.y = row + i;

		/* Iterate over the columns */
		for (int j = -1; j <= 1; j++) {
			coord.x = column + j;

			/* Read value pixel from the image */
			pixel = read_imagef(src_image, sampler, coord);

			/* Accumulate weighted sum */
			sum.xyz += pixel.xyz * sharpeningFilter[filter_index++];
		}
	}

	/* Write new pixel value to output */
	coord = (int2)(column, row);
	write_imagef(dst_image, coord, sum);
}

__kernel void increaseColorSaturation(read_only image2d_t src_image, write_only image2d_t dst_image) {
	/* Get pixel coordinate */
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	/* Read pixel value */
	float4 pixel = read_imagef(src_image, sampler, coord);

	float rgbRed = pixel.x / 255.0f;
	float rgbGreen = pixel.y / 255.0f;
	float rgbBlue = pixel.z / 255.0f;

	float hsvh, hsvs, hsvv;

	float rgbMax = fmax(fmax(rgbRed, rgbGreen), rgbBlue);
	float rgbMin = fmin(fmin(rgbRed, rgbGreen), rgbBlue);

	float MaxMinDiff = rgbMax - rgbMin;

	if (MaxMinDiff == 0)
		hsvh = 0;
	else if (rgbMax == rgbRed)
		hsvh = fmod(((60 * ((rgbGreen - rgbBlue) / MaxMinDiff)) + 360), (float)360.0);
	else if (rgbMax == rgbGreen)
		hsvh = fmod(((60 * ((rgbBlue - rgbRed) / MaxMinDiff)) + 120), (float)360.0);
	else if (rgbMax == rgbBlue)
		hsvh = fmod(((60 * ((rgbRed - rgbGreen) / MaxMinDiff)) + 240), (float)360.0);

	if (hsvh < 0)
		hsvh = hsvh + 360;

	if (rgbMax == 0)
		hsvs = 0;
	else
		hsvs = (MaxMinDiff / rgbMax) * 100;

	hsvv = rgbMax * 100;

	/* Adjust Saturation */
	float adjS = (pow(hsvs, 0.8f)) / 100;

	/* Converting HSV to RGB */
	float v = hsvv / 100;
	float C = v * adjS;
	float X = C * (1.0f - fabs(fmod(hsvh / 60.0f, 2.0f) - 1.0f));
	float M = v - C;

	if (hsvh >= 0.0f && hsvh < 60.0f) {
		rgbRed = C,
			rgbGreen = X,
			rgbBlue = 0.0f;
	}
	else if (hsvh >= 60.0f && hsvh < 120.0f) {
		rgbRed = X,
			rgbGreen = C,
			rgbBlue = 0.0f;
	}
	else if (hsvh >= 120.0f && hsvh < 180.0f) {
		rgbRed = 0.0f,
			rgbGreen = C,
			rgbBlue = X;
	}
	else if (hsvh >= 180.0f && hsvh < 240.0f) {
		rgbRed = 0.0f,
			rgbGreen = X,
			rgbBlue = C;
	}
	else if (hsvh >= 240.0f && hsvh < 300.0f) {
		rgbRed = X,
			rgbGreen = 0.0f,
			rgbBlue = C;
	}
	else {
		rgbRed = C,
			rgbGreen = 0.0f,
			rgbBlue = X;
	}

	/*Calculate in RGB*/

	/*Red channel*/
	pixel.x = (rgbRed + M) * 255;

	/*Green channel*/
	pixel.y = (rgbGreen + M) * 255;

	/*Blue channel*/
	pixel.z = (rgbBlue + M) * 255;

	/* Write new pixel value to output */
	write_imagef(dst_image, coord, pixel);
}