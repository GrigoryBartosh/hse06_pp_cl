#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <cmath>

float* read(int n, int size) {
    float* a = new float[size * size];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a[size * i + j] = 0;
            if (i < n && j < n) {
                std::cin >> a[size * i + j];
            }
        }
    }
    return a;
}

void write(float* a, int n, int size) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << a[size * i + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        cl::Context context(devices);

        cl::CommandQueue queue(context, devices[0]);

        std::ifstream cl_file("convolution.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file),
                              (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        cl::Program program(context, source);

        size_t const block_size = 16;
        program.build(devices, "-D BLOCK_SIZE=16");

        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);

        int n_src, m;
        std::cin >> n_src >> m;

        int n = ceil(1. * n_src / block_size) * block_size;

        const size_t size_a = n * n;
        const size_t size_b = m * m;

        float* a = read(n_src, n);
        float* b = read(m, m);
        float* c = new float[size_a];

        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * size_a);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(float) * size_b);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * size_a);

        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * size_a, a);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * size_b, b);

        cl::Kernel kernel(program, "convolution");
        cl::KernelFunctor convolve_functor(kernel, queue, cl::NullRange, cl::NDRange(n, n), cl::NDRange(block_size, block_size));

        convolve_functor(dev_a, dev_b, dev_c, (int)n, (int)m);

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * size_a, c);

        write(c, n_src, n);

        delete[] b;
        delete[] a;
        delete[] c;
    }
    catch (cl::Error& e) {
        std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
