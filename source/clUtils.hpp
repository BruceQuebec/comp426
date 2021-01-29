#pragma once

#ifndef CLUTILS_H
#define CLUTILS_H

#include <stdio.h>  
#include <stdlib.h>  
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <thread>
#include <ctype.h>
#include <omp.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <GLM/glm.hpp>
#include <windows.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <GL/glu.h>


using namespace std;

/*openCL part*/

bool isCLExtensionSupported(const char *extension, cl_device_id devices_gpu)
{
	// see if the extension is bogus:
	if (extension == NULL || extension[0] == '\0')
		return false;
	char * where = (char *)strchr(extension, ' ');
	if (where != NULL)
		return false;
	// get the full list of extensions:
	size_t extensionSize;
	clGetDeviceInfo(devices_gpu, CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize);
	char *extensions = new char[extensionSize];
	clGetDeviceInfo(devices_gpu, CL_DEVICE_EXTENSIONS, extensionSize, extensions, NULL);
	for (char * start = extensions; ; )
	{
		where = (char *)strstr((const char *)start, extension);
		if (where == 0)
		{
			delete[] extensions;
			return false;
		}

		char * terminator = where + strlen(extension); // points to what should be the separator
		if (*terminator == ' ' || *terminator == '\0' || *terminator == '\r' || *terminator == '\n')
		{
			delete[] extensions;
			return true;
		}
		start = terminator;
	}
}

cl_context CreateContext(cl_device_id devices_gpu)
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;
	
	// First, select an OpenCL platform to run on.
	// For this example, we simply choose the first available
	// platform. Normally, you would query for all available
	// platforms and select the most appropriate one.
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		cerr << "Failed to find any OpenCL platforms." << endl;
		return NULL;
	}

	// since this is an opengl interoperability program,
	// check if the opengl sharing extension is supported
	// (no point going on if it isn¡¯t):
	// (we need the Device in order to ask, so we can't do it any sooner than right here)
	clGetDeviceIDs(firstPlatformId, CL_DEVICE_TYPE_GPU, 1, &devices_gpu, NULL);
	if (!isCLExtensionSupported("cl_khr_gl_sharing", devices_gpu))
	{
		fprintf(stderr, "cl_khr_gl_sharing is not supported -- sorry.\n");
		return NULL;
	}

	// Next, create an OpenCL context on the platform. Attempt to
	// create a GPU-based context, and if that fails, try to create
	// a CPU-based context.
	// 3. create a special opencl context based on the opengl context:
	cl_context_properties contextProperties[] =
	{
		CL_GL_CONTEXT_KHR,
		(cl_context_properties)wglGetCurrentContext(),
		CL_WGL_HDC_KHR,
		(cl_context_properties)wglGetCurrentDC(),
		CL_CONTEXT_PLATFORM, 
		(cl_context_properties)firstPlatformId,
		0
	};

	/*cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};*/
	
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_ALL, NULL, NULL, &errNum);
	
	if (errNum != CL_SUCCESS)
	{
		cerr <<
			"Failed to create an OpenCL GPU or CPU context.";
		return NULL;
	}
	return context;
}


cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device, cl_device_type my_dev_type)
{
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;
	// First get the size of the devices buffer
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	
	if (errNum != CL_SUCCESS)
	{
		cerr << "Failed call to clGetContextInfo(..., GL_CONTEXT_DEVICES, ...)";
		return NULL;
	}
	if (deviceBufferSize <= 0)
	{
		cerr << "No devices available.";
		return NULL;
	}
	// Allocate memory for the devices buffer
	int num_devices = deviceBufferSize / sizeof(cl_device_id);
	devices = new cl_device_id[num_devices];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Failed to get device IDs";
		return NULL;
	}

	// In this example, we just choose the first available device.
	// In a real program, you would likely use all available
	// devices or choose the highest performance device based on
	// OpenCL device queries.
	cl_device_type dev_type;
	cl_int dev_cu;
	for (int i = 0; i < num_devices; ++i) {
		clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
		clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(dev_cu), &dev_cu, NULL);
		if (my_dev_type == CL_DEVICE_TYPE_GPU && my_dev_type == my_dev_type && dev_cu == 16) {
			commandQueue = clCreateCommandQueue(context, devices[i], 0, NULL);
			*device = devices[i];
		}
		else if (my_dev_type == CL_DEVICE_TYPE_CPU && my_dev_type == my_dev_type) {
			commandQueue = clCreateCommandQueue(context, devices[i], 0, NULL);
			*device = devices[i];
		}
		if (commandQueue == NULL)
		{
			cerr << "Failed to create commandQueue for device " + i;
			return NULL;
		}
	}
	delete[] devices;
	return commandQueue;
}

cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName, string includefile) {
	cl_int errNum;
	cl_program program;
	ifstream kernelFile(fileName, ios::in);
	if (!kernelFile.is_open())
	{
		cerr << "Failed to open file for reading: " << fileName << endl;
		return NULL;
	}
	ostringstream oss;
	oss << kernelFile.rdbuf();
	string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, NULL);
	if (program == NULL)
	{
		cerr << "Failed to create CL program from source." << endl;
		return NULL;
	}
	if (includefile != "") {
		errNum = clBuildProgram(program, 0, NULL, includefile.c_str(), NULL, NULL);
	}
	else {
		errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	}

	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
		cerr << "Error in kernel: " << endl;
		cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}
	return program;
}

bool CreateMemObjects(cl_context context, cl_mem memObjects[3], float *a, float *b, const int ARRAY_SIZE)
{
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, a, NULL);
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, b, NULL);
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ARRAY_SIZE, NULL, NULL);
	if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
	{
		cerr << "Error creating memory objects." << endl;
		return false;
	}
	return true;
}

bool CreateMemObjectsForCellStatusUpdateGPU(cl_context context, cl_mem memObjects[6], int *host_cell, int *host_medDir, int* num_rows, int* num_per_row, const int ARRAY_SIZE)
{
	cl_int* err = NULL;
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, host_cell, err);
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, host_medDir, err);
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), num_rows, err);
	memObjects[3] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), num_per_row, err);
	memObjects[4] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * ARRAY_SIZE, NULL, err);
	memObjects[5] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * ARRAY_SIZE, NULL, err);

	if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL || memObjects[3] == NULL || memObjects[4] == NULL || memObjects[5] == NULL)
	{
		cerr << "Error creating memory objects." << endl;
		return false;
	}
	return true;
}

bool CreateMemObjectsForCelldisplayCPU(cl_context context, cl_mem memObjects[2], int *host_cell, int* num_per_row, const int ARRAY_SIZE)
{
	cl_int* err = NULL;
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, host_cell, err);
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, num_per_row, err);

	if (memObjects[0] == NULL || memObjects[1] == NULL)
	{
		cerr << "Error creating memory objects." << endl;
		return false;
	}
	return true;
}

void safeQuit(cl_context context, cl_kernel Kernel, cl_program Program, cl_command_queue cmdQueue, cl_mem dPobj, cl_mem dCobj)
{
	clReleaseContext(context);
	clReleaseKernel(Kernel);
	clReleaseProgram(Program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(dPobj);
	exit(0);
}



#endif