#include <common/Common.h>

// Block wise reduction (addtion operator) using the shared memory
template<typename T>
__device__ T ReduceBlock( T val, volatile T* cache )
{
	cache[threadIdx.x] = val;
	// Make sure that the data has been written to the shared memory
	__syncthreads(); 
	for( int i = 1; i < blockDim.x; i <<= 1 )
	{
		if( threadIdx.x & i )
			cache[threadIdx.x] += cache[threadIdx.x ^ i];
		__syncthreads(); // Make sure that the data has been updated
	}
	// Return the last entry in the cache as a result
	return cache[blockDim.x - 1];
}

// Warp-wise reduction (addtion operator) using the shuffle instruction
template<typename T>
__device__ T ReduceWarp( T val )
{
	for( int i = 1; i < warpSize; i <<= 1 )
	{
		// Read the register of the corresponding thread
		T tmp = __shfl_xor( val, i );
		// Add it to the current value
		val += tmp;
	}
	// Return value of the last thread in the warp
	return __shfl( val, warpSize - 1 );
}

extern "C" __global__ void ReduceBlockKernel( u32 size, const int* input, int* output )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;

	int val = 0;
	if( index < size ) val = input[index];

	// Shared memory cache
	extern __shared__ int cache[];
	// Block-wise reduction
	val = ReduceBlock( val, cache );
	// Atomically add the block sum to the global counter
	if( threadIdx.x == 0 ) atomicAdd( output, val );
}

extern "C" __global__ void ReduceWarpKernel( u32 size, const int* input, int* output )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	u32 laneIndex = threadIdx.x & ( warpSize - 1 );

	int val = 0;
	if( index < size ) val = input[index];
	// Warp-wise reduction
	val = ReduceWarp( val );
	// Atomically add the warp sum to the global counter
	if( laneIndex == 0 ) atomicAdd( output, val );
}
