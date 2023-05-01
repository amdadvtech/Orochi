#include <common/Common.h>

extern "C" __global__ void HistogramKernel( u32 size, u32 threads, u32 bins, const int* input, int* output, int* counter )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	u32 laneIndex = threadIdx.x & ( warpSize - 1 );

	u32 warpsPerBlock = ( blockDim.x + warpSize - 1 ) / warpSize;
	u32 warps = gridDim.x * warpsPerBlock;
	u32 warpIndex = threadIdx.x / warpSize + blockIdx.x * warpsPerBlock;

	for( u32 i = index; i < bins; i += threads )
		output[i] = 0;

	if( laneIndex == 0 )
	{
		atomicAdd( counter, 1 );
		while( atomicAdd( counter, 0 ) < warps )
			;
	}
	__syncthreads();

	__threadfence();
	for( u32 i = index; i < size; i += threads )
	{
		int val = input[i];
		atomicAdd( &output[val], 1 );
	}
}
