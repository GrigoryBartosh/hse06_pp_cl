#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

size_t const BLOCK_SIZE = 256;

float* read(int n, int size) {
    float* a = new float[size];
    for (int i = 0; i < size; i++) {
        a[i] = 0;
        if (i < n) {
            std::cin >> a[i];
        }
    }
    return a;
}

void write(float* a, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

void block_copy(float* from, size_t n, float* &to, size_t m, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * n);
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * m);

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * n, from);

    size_t size = ceil(1. * n / BLOCK_SIZE) * BLOCK_SIZE;

    cl::Kernel kernel(program, "block_copy");

    cl::KernelFunctor convolve_functor(kernel, queue, cl::NullRange, cl::NDRange(size), cl::NDRange(BLOCK_SIZE));
    convolve_functor(dev_input, dev_output, static_cast<unsigned int>(n), static_cast<unsigned int>(m));

    to[0] = 0.0;
    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * m, to);
}

void block_add(float* from, size_t n, float* &to, size_t m, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_input_partial(context, CL_MEM_READ_ONLY, sizeof(float) * n);
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * m);
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * m);

    queue.enqueueWriteBuffer(dev_input_partial, CL_TRUE, 0, sizeof(float) * n, from);
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * m, to);

    size_t size = ceil(1. * m / BLOCK_SIZE) * BLOCK_SIZE;

    cl::Kernel kernel(program, "block_add");

    cl::KernelFunctor convolve_functor(kernel, queue, cl::NullRange, cl::NDRange(size), cl::NDRange(BLOCK_SIZE));
    convolve_functor(dev_input_partial, dev_input, dev_output, static_cast<unsigned int>(m));

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * m, to);
}

void prefix_sum(float* res, size_t n, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * n);
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * n, res);

    size_t size = ceil(1. * n / BLOCK_SIZE) * BLOCK_SIZE;

    cl::Kernel kernel(program, "prefix_sum");
    cl::KernelFunctor convolve_functor(kernel, queue, cl::NullRange, cl::NDRange(size), cl::NDRange(BLOCK_SIZE));

    convolve_functor(dev_input, dev_output, cl::__local(sizeof(float) * BLOCK_SIZE), cl::__local(sizeof(float) * BLOCK_SIZE), static_cast<unsigned int>(n));

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * n, res);

    if (n > BLOCK_SIZE) {
        size_t m = ceil(1. * n / BLOCK_SIZE);
        float* sums = new float[m];

        block_copy(res, n, sums, m, context, program, queue);
        prefix_sum(sums, m, context, program, queue);
        block_add(sums, m, res, n, context, program, queue);

        delete[] sums;
    }
}

int main()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);

        cl::Context context(devices);

        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        std::ifstream cl_file("prefix_sum.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        cl::Program program(context, source);
        program.build(devices);

        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);

        size_t n_src = 0;
        std::cin >> n_src;

        size_t n = ceil(1. * n_src / BLOCK_SIZE) * BLOCK_SIZE;

        float* a = read(n_src, n);

        prefix_sum(a, n, context, program, queue);

        write(a, n_src);

        delete[] a;
    }
    catch (cl::Error const & e) {
        std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}