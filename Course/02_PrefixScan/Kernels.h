#include <common/Common.h>

enum
{
	HillisSteele = 0,
	Blelloch = 1
};

template<typename T>
__device__ T ReduceBlock( T val, volatile T* cache )
{
	cache[threadIdx.x] = val;
	__syncthreads();
	int active = blockDim.x >> 1;
	for( int i = 1; i < blockDim.x; i <<= 1 )
	{
		if( threadIdx.x < active )
		{
			int leftIndex = i * ( 2 * threadIdx.x + 1 ) - 1;
			int rightIndex = i * ( 2 * threadIdx.x + 2 ) - 1;
			cache[rightIndex] += cache[leftIndex];
		}
		active >>= 1;
		__syncthreads();
	}
	return cache[blockDim.x - 1];
}

template<typename T>
__device__ T ScanBlockBlelloch( T val, volatile T* cache )
{
	ReduceBlock( val, cache );
	if( threadIdx.x == 0 )
		cache[blockDim.x - 1] = static_cast<T>( 0 );
	__syncthreads();
	int active = 1;
	for( int i = blockDim.x >> 1; i >= 1; i >>= 1 )
	{
		if( threadIdx.x < active )
		{
			int leftIndex = i * ( 2 * threadIdx.x + 1 ) - 1;
			int rightIndex = i * ( 2 * threadIdx.x + 2 ) - 1;
			T tmp = cache[rightIndex];
			cache[rightIndex] += cache[leftIndex];
			cache[leftIndex] = tmp;
		}
		active <<= 1;
		__syncthreads();
	}
	return cache[threadIdx.x] + val;
}

template<typename T>
__device__ T ScanBlockHillisSteele( T val, volatile T* cache )
{
	cache[threadIdx.x] = val;
	__syncthreads();
	for( int i = 1; i < blockDim.x; i <<= 1 )
	{
		if( threadIdx.x >= i ) val += cache[threadIdx.x - i];
		__syncthreads();
		cache[threadIdx.x] = val;
		__syncthreads();
	}
	return cache[threadIdx.x];
}

template<int BlockScan, typename T>
__device__ T ScanDevice( T val, volatile T* cache, T* sum, int* counter )
{
	if constexpr( BlockScan == HillisSteele ) 
		val = ScanBlockHillisSteele( val, cache );
	else
		val = ScanBlockBlelloch( val, cache );
	__shared__ T offset;
	if( threadIdx.x == blockDim.x - 1 )
	{
		while( atomicAdd( counter, 0 ) < blockIdx.x )
			;
		offset = atomicAdd( sum, val );
		atomicAdd( counter, 1 );
	}
	__syncthreads();
	return offset + val;
}

extern "C" __global__ void ScanDeviceHillisSteeleKernel( u32 size, const int* input, int* output, int* sum, int* counter )
{
	const u32 index = threadIdx.x + blockDim.x * blockIdx.x;

	int val = 0;
	if( index < size ) val = input[index];

	extern __shared__ int cache[];
	val = ScanDevice<HillisSteele>( val, cache, sum, counter );

	if( index < size ) output[index] = val;
}

extern "C" __global__ void ScanDeviceBlellochKernel( u32 size, const int* input, int* output, int* sum, int* counter )
{
	const u32 index = threadIdx.x + blockDim.x * blockIdx.x;

	int val = 0;
	if( index < size ) val = input[index];

	extern __shared__ int cache[];
	val = ScanDevice<Blelloch>( val, cache, sum, counter );

	if( index < size ) output[index] = val;
}
