#include <common/Common.h>

// A kernel computing a histogram of input values
extern "C" __global__ void HistogramKernel( u32 size, u32 threads, u32 bins, const int* input, int* output, int* counter )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	u32 laneIndex = threadIdx.x & ( warpSize - 1 );

	// The number of warps per block
	u32 warpsPerBlock = ( blockDim.x + warpSize - 1 ) / warpSize;
	// The total number of warps 
	u32 warps = gridDim.x * warpsPerBlock;

	// Reset the histogram
	for( u32 i = index; i < bins; i += threads )
		output[i] = 0;

	// Global barrier
	if( laneIndex == 0 )
	{
		atomicAdd( counter, 1 );
		// Keep spinning until all warps are processed
		while( atomicAdd( counter, 0 ) < warps )
			;
	}
	// Wait for the spinning first thread in the warp
	__syncthreads();

	// We wrote to the global memory (resetting the histogram)
	// Thus, flush the caches
	__threadfence();

	// Compute the histogram
	for( u32 i = index; i < size; i += threads )
	{
		int val = input[i];
		atomicAdd( &output[val], 1 );
	}
}
