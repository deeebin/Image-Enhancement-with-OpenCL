#define CL_USE_DEPRECATED_OPENCL_2_0_APIS	// using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS				// enable OpenCL exemptions
#pragma comment(lib, "OpenCL.lib")
// C++ standard library and STL headers
#include <iostream>
#include <vector>
#include <fstream>

// OpenCL header, depending on OS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

cl_ulong startTime, endTime;

#include "common.h"
#include "bmpfuncs.h"

using namespace std;
using namespace cl;

int main(void)
{
	Platform platform;			// device's platform
	Device device;				// device used
	Context context;			// context for the device
	Program program;			// OpenCL program object
	CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	unsigned char* inputImage;
	unsigned char* outputImage;
	unsigned char* outputImage2;
	unsigned char* outputImage3;
	unsigned char* outputImage4;
	int imgWidth, imgHeight, imageSize;
	Event timeEvent, timeEvent2, timeEvent3, timeEvent4, timeEvent5, timeEvent6;

	ImageFormat imgFormat;
	Image2D inputImgBuffer, inputImgBuffer2, inputImgBuffer3, inputImgBuffer4;
	Image2D outputImgBuffer, outputImgBuffer2, outputImgBuffer3, outputImgBuffer4;

	try {
		// select an OpenCL device
		if (!select_one_device(&platform, &device))
		{
			// if no device selected
			quit_program("Device not selected.");
		}

		// create a context from device
		context = Context(device);

		// build the program
		if (!build_program(&program, &context, "kernel.cl"))
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create kernels
		vector<Kernel> allKernels;
		program.createKernels(&allKernels);

		// create command queue
		queue = CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

		// prompt input image
		string fileName;
		cout << "Enter filename : ";
		getline(cin, fileName);

		ifstream readFile;
		readFile.open(fileName);
		cout << "--------------------" << endl;

		// If file is not found
		if (!readFile.good())
		{
			readFile.close();
			cout << "'" << fileName << "' is not found." << endl;
			cout << "--------------------" << endl;
			return 1;
		}

		// read input image
		inputImage = read_BMP_RGB_to_RGBA(fileName.c_str(), &imgWidth, &imgHeight);

		// allocate memory for output image
		imageSize = imgWidth * imgHeight * 4;
		outputImage = new unsigned char[imageSize];
		outputImage2 = new unsigned char[imageSize];
		outputImage3 = new unsigned char[imageSize];
		outputImage4 = new unsigned char[imageSize];

		// image format
		imgFormat = ImageFormat(CL_RGBA, CL_UNORM_INT8);

		// create image objects
		inputImgBuffer = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)inputImage);
		outputImgBuffer = Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);

		vector<cl_int> histogram(256, 0); // Initialize value 0
		Buffer histogramBuffer = Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * 256);

		// set kernel arguments
		allKernels[0].setArg(0, inputImgBuffer);
		allKernels[0].setArg(1, histogramBuffer);

		// enqueue kernel
		NDRange offset(0, 0);
		NDRange globalSize(imgWidth, imgHeight);

		queue.enqueueNDRangeKernel(allKernels[0], offset, globalSize, NullRange, NULL, &timeEvent);

		cout << "Kernel calculateHistogram enqueued." << endl;
		cout << "--------------------" << endl;

		queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, sizeof(cl_int) * 256, &histogram[0]);

		startTime = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		endTime = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		//Calculate the time total
		cl_ulong timetotal = endTime - startTime;

		cout << "Start time: " << startTime << " nanoseconds" << endl;
		cout << "End time: " << endTime << " nanoseconds" << endl;
		cout << "Read time: " << timetotal / 1000000.0f << " nanoseconds" << endl;
		cout << "--------------------" << endl;

		histogramBuffer = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 256, &histogram[0]);

		// set kernel arguments
		allKernels[1].setArg(0, inputImgBuffer);
		allKernels[1].setArg(1, outputImgBuffer);
		allKernels[1].setArg(2, histogramBuffer);

		queue.enqueueNDRangeKernel(allKernels[1], offset, globalSize, NullRange, NULL, &timeEvent2);

		cout << "Kernel HistogramEqualization enqueued." << endl;
		cout << "--------------------" << endl;

		// enqueue command to read image from device to host memory
		cl::size_t<3> origin, region;
		origin[0] = origin[1] = origin[2] = 0;
		region[0] = imgWidth;
		region[1] = imgHeight;
		region[2] = 1;

		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		startTime = timeEvent2.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		endTime = timeEvent2.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		//Calculate the time total
		timetotal = endTime - startTime;

		cout << "Start time: " << startTime << " nanoseconds" << endl;
		cout << "End time: " << endTime << " nanoseconds" << endl;
		cout << "Read time: " << timetotal / 1000000.0f << " nanoseconds" << endl;
		cout << "--------------------" << endl;

		inputImgBuffer2 = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
		outputImgBuffer2 = Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage2);

		// set kernel arguments
		allKernels[2].setArg(0, inputImgBuffer2);
		allKernels[2].setArg(1, outputImgBuffer2);

		queue.enqueueNDRangeKernel(allKernels[2], offset, globalSize, NullRange, NULL, &timeEvent3);

		cout << "Kernel increaseSharpness enqueued." << endl;
		cout << "--------------------" << endl;

		queue.enqueueReadImage(outputImgBuffer2, CL_TRUE, origin, region, 0, 0, outputImage2);

		startTime = timeEvent3.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		endTime = timeEvent3.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		//Calculate the time total
		timetotal = endTime - startTime;

		cout << "Start time: " << startTime << " nanoseconds" << endl;
		cout << "End time: " << endTime << " nanoseconds" << endl;
		cout << "Read time: " << timetotal / 1000000.0f << " nanoseconds" << endl;
		cout << "--------------------" << endl;

		inputImgBuffer3 = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage2);
		outputImgBuffer3 = Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage3);

		// set kernel arguments
		allKernels[2].setArg(0, inputImgBuffer3);
		allKernels[2].setArg(1, outputImgBuffer3);

		queue.enqueueNDRangeKernel(allKernels[2], offset, globalSize, NullRange, NULL, &timeEvent4);

		cout << "Kernel increaseSharpness enqueued." << endl;
		cout << "--------------------" << endl;

		queue.enqueueReadImage(outputImgBuffer3, CL_TRUE, origin, region, 0, 0, outputImage3);

		startTime = timeEvent4.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		endTime = timeEvent4.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		//Calculate the time total
		timetotal = endTime - startTime;

		cout << "Start time: " << startTime << " nanoseconds" << endl;
		cout << "End time: " << endTime << " nanoseconds" << endl;
		cout << "Read time: " << timetotal / 1000000.0f << " nanoseconds" << endl;
		cout << "--------------------" << endl;

		inputImgBuffer4 = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage3);
		outputImgBuffer4 = Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage4);

		// set kernel arguments
		allKernels[3].setArg(0, inputImgBuffer4);
		allKernels[3].setArg(1, outputImgBuffer4);

		queue.enqueueNDRangeKernel(allKernels[3], offset, globalSize, NullRange, NULL, &timeEvent5);

		cout << "Kernel increaseColorSaturation enqueued." << endl;
		cout << "--------------------" << endl;

		queue.enqueueReadImage(outputImgBuffer4, CL_TRUE, origin, region, 0, 0, outputImage4);

		startTime = timeEvent5.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		//Set a stop event to take the time
		endTime = timeEvent5.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		//Calculate the time total
		timetotal = endTime - startTime;

		cout << "Start time: " << startTime << " nanoseconds" << endl;
		cout << "End time: " << endTime << " nanoseconds" << endl;
		cout << "Read time: " << timetotal / 1000000.0f << " nanoseconds" << endl;
		cout << "--------------------" << endl;

		// output results to image file
		write_BMP_RGBA_to_RGB("output.bmp", outputImage4, imgWidth, imgHeight);

		cout << "Image " << fileName << " is enhanced and saved as (output.bmp)." << endl;

		// deallocate memory
		free(inputImage);
		free(outputImage);
		free(outputImage2);
		free(outputImage3);
		free(outputImage4);
	}
	// catch any OpenCL function errors
	catch (Error e) {
		// call function to handle errors
		handle_error(e);
	}
	system("pause");
	return 0;
}