__kernel void prefix_sum(__global float * input, __global float * output, __local float * a, __local float * b, uint size) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);

    if (gid < size) {
        a[lid] = input[gid];
        b[lid] = input[gid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 1; i < block_size; i *= 2) {
        b[lid] = a[lid];
        if(lid >= i) {
            b[lid] += a[lid - i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        __local float* t = a;
        a = b;
        b = t;
    }

    if (gid < size) {
        output[gid] = a[lid];
    }
}

__kernel void block_copy(__global float *input, __global float *output, uint input_size, uint output_size) {
    uint gid = get_global_id(0);
    uint block_size = get_local_size(0);

    uint ind = gid / block_size + 1;

    if (gid < input_size && ind < output_size && 1 + gid == ind * block_size) {
        output[ind] = input[gid];
    }
}

__kernel void block_add(__global float *partial_input, __global float *input, __global float *output, uint size) {
    uint gid = get_global_id(0);
    uint block_size = get_local_size(0);

    if (gid < size) {
        output[gid] = input[gid] + partial_input[gid / block_size];
    }
}