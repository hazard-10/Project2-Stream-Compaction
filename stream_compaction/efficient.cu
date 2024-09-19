#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upSweepKernel(int n, int d, int *data){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n){
                return;
            }
            if ((index + 1) % (1 << (d + 1)) == 0){
                data[index] += data[index - (1 << d)];
            }
        }

        __global__ void downSweepKernel(int n, int d, int *data){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n){
                return;
            }
            if ((index + 1) % (1 << (d + 1)) == 0){
                int root = data[index];
                int left_index = index - (1 << d);
                data[index] += data[left_index];
                data[left_index] = root;
            }

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int d_round = ilog2ceil(n);
            int full_size = 1 << d_round;
            int block_size = 256;
            dim3 fullBlocksPerGrid((block_size + full_size - 1) / block_size);

            int *d_data;
            cudaMalloc((void **)&d_data, full_size * sizeof(int));
            cudaMemset(d_data, 0, full_size * sizeof(int));
            cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            // Up-sweep
            for (int d = 0; d < d_round; d++){
                upSweepKernel<<<fullBlocksPerGrid, block_size>>>(full_size, d, d_data);
            }
            // Down-sweep
            cudaMemset(d_data + full_size - 1, 0, sizeof(int));
            for (int d = d_round - 1; d >= 0; d--){
                downSweepKernel<<<fullBlocksPerGrid, block_size>>>(full_size, d, d_data);
            }
            timer().endGpuTimer();
            // Copy result to odata
            cudaMemcpy(odata, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
