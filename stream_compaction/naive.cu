#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void naiveScanKernel(int n, int offset, int *odata, const int *idata){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n){
                return;
            }

            if (index >= offset){
                odata[index] = idata[index - offset] + idata[index];
            }
            else{
                odata[index] = idata[index];
            }
        }

         __global__ void naiveScanFirstRound(int n, int *odata, const int *idata){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n){
                return;
            }
            if (index == 0){
                odata[index] = 0;
            }
            else if (index == 1){
                odata[index] = idata[index - 1];
            }
            else{
                odata[index] = idata[index - 1] + idata[index - 2];
            }
         }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata){
            int blockSize = 256;
            dim3 fullBlocksPerGrid((blockSize + n - 1) / blockSize);

            // TODO
            int d_round = ilog2ceil(n);
            int *dstFirst;
            int *dstSecond;
            cudaMalloc((void **)&dstFirst, n * sizeof(int));
            cudaMalloc((void **)&dstSecond, n * sizeof(int));

            cudaMemcpy(dstFirst, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            naiveScanFirstRound<<<fullBlocksPerGrid, blockSize>>>(n, dstSecond, dstFirst);
            std::swap(dstFirst, dstSecond);

            for (int d = 1; d < d_round; d++){
                int d_offset = 1 << d; // 2, 4, 8
                naiveScanKernel<<<fullBlocksPerGrid, blockSize>>>(n, d_offset, dstSecond, dstFirst);
                std::swap(dstFirst, dstSecond);
            }
            timer().endGpuTimer();
            // setFirstAsZero<<<1, 1>>>(dstFirst);
            cudaMemcpy(odata, dstFirst, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dstFirst);
            cudaFree(dstSecond);
        }
    }
}
