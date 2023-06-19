#include <common/Common.h>

enum
{
	HillisSteele = 0,
	Blelloch = 1
};

// Up-sweep pass (addtion operator) of the Blelloch's algorithm
template<typename T>
__device__ void ReduceBlock( T val, volatile T* cache )
{
	cache[threadIdx.x] = val;
	__syncthreads();
	int active = blockDim.x >> 1;
	for( int i = 1; i < blockDim.x; i <<= 1 )
	{
		if( threadIdx.x < active )
		{
			// This access pattern must match the down-sweep pahse
			int leftIndex = i * ( 2 * threadIdx.x + 1 ) - 1;
			int rightIndex = i * ( 2 * threadIdx.x + 2 ) - 1;
			cache[rightIndex] += cache[leftIndex];
		}
		active >>= 1;
		__syncthreads(); // Synchronize after each iteration
	}
}

// Block-wise prefix scan (addtion operator) implementing the Blelloch's algorithm using shared memory
template<typename T>
__device__ T ScanBlockBlelloch( T val, volatile T* cache )
{
	// Up-sweep phase (reduction)
	ReduceBlock( val, cache );
	// Assign an identity element to the root ('0' for addition)
	if( threadIdx.x == 0 )
		cache[blockDim.x - 1] = static_cast<T>( 0 );
	__syncthreads(); // Make sure the assignment has been finished
	int active = 1;
	for( int i = blockDim.x >> 1; i >= 1; i >>= 1 )
	{
		if( threadIdx.x < active )
		{
			// The same access pattern as in the up-sweep phase
			int leftIndex = i * ( 2 * threadIdx.x + 1 ) - 1;
			int rightIndex = i * ( 2 * threadIdx.x + 2 ) - 1;
			// Save the right child value
			T tmp = cache[rightIndex];
			// Assign the sum of the left child and parent to the right child
			cache[rightIndex] += cache[leftIndex];
			// Assign the original right child value to the left child
			cache[leftIndex] = tmp;
		}
		active <<= 1;
		__syncthreads();  // Synchronize after each iteration
	}
	// Each thread returns the corresponding prefix scan value
	return cache[threadIdx.x] + val;
}

// Block-wise prefix (addtion operator) scan implementing Hillis-Steele algorithm using shared memory
template<typename T>
__device__ T ScanBlockHillisSteele( T val, volatile T* cache )
{
	cache[threadIdx.x] = val;
	// Make sure that the data has been written to the shared memory
	__syncthreads();
	for( int i = 1; i < blockDim.x; i <<= 1 )
	{
		// Add the value to the local variable
		if( threadIdx.x >= i ) val += cache[threadIdx.x - i];
		__syncthreads(); // Make sure that all threads have read the data
		// Write the update value back to the cache
		cache[threadIdx.x] = val;
		__syncthreads(); // Make sure that the data has been updated
	}
	// Each thread returns the corresponding prefix scan value
	return cache[threadIdx.x];
}

// Device-wise prefix scan (addtion operator) using the spinlock
template<int BlockScan, typename T>
__device__ T ScanDevice( T val, volatile T* cache, T* sum, int* counter )
{
	// Block-wise prefix scan
	if constexpr( BlockScan == HillisSteele ) 
		val = ScanBlockHillisSteele( val, cache );
	else
		val = ScanBlockBlelloch( val, cache );
	// The block offset shared with other threads
	__shared__ T offset;
	// Let the last thread to computeth the block offset
	if( threadIdx.x == blockDim.x - 1 )
	{
		// Spinlock => wait until all previous blocks are processed
		while( atomicAdd( counter, 0 ) < blockIdx.x )
			;
		// Add the block sum to obtain the block offset
		offset = atomicAdd( sum, val );
		// Use memory fence to ensure the order of atomics
		__threadfence();
		// Signalize the next block that can be processed
		atomicAdd( counter, 1 );
	}
	// Wait for the last thread
	__syncthreads();
	// The result is the sum of block offset and the block prefix scan
	return offset + val;
}

extern "C" __global__ void ScanDeviceHillisSteeleKernel( u32 size, const int* input, int* output, int* sum, int* counter )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;

	int val = 0;
	if( index < size ) val = input[index];

	// Shared memory cache
	extern __shared__ int cache[];
	// Device-wise scan
	val = ScanDevice<HillisSteele>( val, cache, sum, counter );
	// Write the result to the output buffer
	if( index < size ) output[index] = val;
}

extern "C" __global__ void ScanDeviceBlellochKernel( u32 size, const int* input, int* output, int* sum, int* counter )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;

	int val = 0;
	if( index < size ) val = input[index];

	// Device-wise scan
	extern __shared__ int cache[];
	// Device-wise scan
	val = ScanDevice<Blelloch>( val, cache, sum, counter );
	// Write the result to the output buffer
	if( index < size ) output[index] = val;
}
