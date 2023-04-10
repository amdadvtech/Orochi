#include <common/Common.h>

template<typename T>
__device__ T ReduceBlock( T val, volatile T* cache )
{
	cache[threadIdx.x] = val;
	__syncthreads();
	int active = blockDim.x;
	for( int i = 1; i < blockDim.x; i <<= 1 )
	{
		#if 0
		if( ( threadIdx.x & i ) == i )
			cache[threadIdx.x] += cache[threadIdx.x ^ i];
		#else
		active >>= 1;
		if( threadIdx.x < active)
			cache[i * ( 2 * threadIdx.x + 2 ) - 1] += cache[i * ( 2 * threadIdx.x + 1 ) - 1];
		#endif
		__syncthreads();
	}
	return cache[blockDim.x - 1];
}

template<typename T>
__device__ T ReduceWarp( T val )
{
	const u32 laneIndex = threadIdx.x & ( warpSize - 1 );
	for( int i = 1; i < warpSize; i <<= 1 )
	{
		T tmp = __shfl_xor( val, i );
		val += tmp;
	}
	return __shfl( val, warpSize - 1 );
}

extern "C" __global__ void ReduceBlockKernel( u32 size, const int* input, int* output )
{
	const u32 index = threadIdx.x + blockDim.x * blockIdx.x;

	int val = 0;
	if( index < size ) val = input[index];

	extern __shared__ int cache[];
	val = ReduceBlock( val, cache );
	if( threadIdx.x == 0 ) atomicAdd( output, val );
}

extern "C" __global__ void ReduceWarpKernel( u32 size, const int* input, int* output )
{
	const u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	const u32 laneIndex = threadIdx.x & ( warpSize - 1 );

	int val = 0;
	if( index < size ) val = input[index];
	
	val = ReduceWarp( val );
	if( laneIndex == 0 ) atomicAdd( output, val );
}
