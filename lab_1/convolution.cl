__kernel void convolution(__global float * a, __global float * b, __global float * c, int n, int m)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row >= n || col >= n) {
        return;
    }

    float res = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            int y = row + i - m / 2;
            int x = col + j - m / 2;

            if (y >= 0 && y < n && x >= 0 && x < n) {
                res += a[y * n + x] * b[i * m + j];
            }
        }
    }

    c[row * n + col] = res;
}