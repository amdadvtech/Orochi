#include <common/Common.h>

// Warp-wise inclusive prefix scan (addition operator) using the shuffle instruction
template<typename T>
__device__ T ScanWarp( T val )
{
	int laneIndex = threadIdx.x & ( warpSize - 1 );
	for( int i = 1; i < warpSize; i <<= 1 )
	{
		// Read the register of the corresponding thread
		T tmp = __shfl_up( val, i );
		// Add it to the current value
		if( laneIndex >= i ) val += tmp;
	}
	// Each thread returns the corresponding prefix scan value
	return val;
}

// Warp-wise exclusive prefix scan with binary values (addition operator) using the ballot instruction
__device__ int ScanWarpBinary( bool val )
{
	int laneIndex = threadIdx.x & ( warpSize - 1 );
	// Read the data of other threads
	u64 ballot = __ballot( val );
	// Mask out the higher bits (values of threads with greater lane index)
	// Use popcount to compute the number of bits set to one to ge the final result
	return __popcll( ballot & ( ( 1ull << laneIndex ) - 1ull ) );
}

// Naive enqueue kernel using atomic add to get the offset
extern "C" __global__ void EnqueueNaiveKernel( u32 size, const int* input, int* output, int* counter )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;

	int val = 0;
	if( index < size ) val = input[index];

	// Predicate indicating the prefix scan input (binary)
	// Indicating whether we enqueue the input value or not
	bool enqueue = val & 1;
	// We use just atomic add to get the offset
	// Not efficient as it is called per thread
	if( index < size && enqueue ) output[atomicAdd( counter, 1 )] = val;
}

// Enqueue kernel using the warp-wise prefix scan
extern "C" __global__ void EnqueueKernel( u32 size, const int* input, int* output, int* counter )
{
	u32 index0 = threadIdx.x + blockDim.x * blockIdx.x;
	u32 index1 = index0 + gridDim.x * blockDim.x;
	u32 laneIndex = threadIdx.x & ( warpSize - 1 );

	int val0 = 0;
	int val1 = 0;
	if( index0 < size ) val0 = input[index0];
	if( index1 < size ) val1 = input[index1];

	// Predicate indicating the prefix scan input
	// Indicating whether we enqueue the input value or not
	bool enqueue0 = val0 & 1;
	bool enqueue1 = val1 & 1;
	u32 enqueuedCount = enqueue0 + enqueue1;

	// Warp-wise prefix scan
	// We subtract the enqueue value to get the exclusive prefix scan
	int warpScan = ScanWarp( enqueuedCount ) - enqueuedCount;

	// The last thread in the warp atomically adds it value to the global counter to get the offset
	// The value of the last thread is equal to the sum of all elements in the warp
	int warpOffset = 0;
	if( laneIndex == warpSize - 1 ) warpOffset = atomicAdd( counter, warpScan + enqueuedCount );

	// We exchange the offset of the warp using the shuffle instruction
	warpOffset = __shfl( warpOffset, warpSize - 1 );

	u32 offset = warpOffset + warpScan;
	if( index0 < size && enqueue0 ) output[offset++] = val0;
	if( index1 < size && enqueue1 ) output[offset] = val1;
}

// Enqueue kernel using the binary warp-wise prefix scan
extern "C" __global__ void EnqueueBinaryKernel( u32 size, const int* input, int* output, int* counter )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	u32 laneIndex = threadIdx.x & ( warpSize - 1 );

	int val = 0;
	if( index < size ) val = input[index];

	// Predicate indicating the prefix scan input (binary)
	// Indicating whether we enqueue the input value or not
	bool enqueue = val & 1;

	// Binary warp-wise prefix scan
	int warpScan = ScanWarpBinary( enqueue );

	// The last thread in the warp atomically adds it value to the global counter to get the offset
	// The value of the last thread is equal to the sum of all elements in the warp
	int warpOffset = 0;
	if( laneIndex == warpSize - 1 ) warpOffset = atomicAdd( counter, warpScan + enqueue );

	// We exchange the offset of the warp using the shuffle instruction
	warpOffset = __shfl( warpOffset, warpSize - 1 );

	if( index < size && enqueue ) output[warpOffset + warpScan] = val;
}

// Enqueue kernel using the binary warp-wise prefix scan exploiting the complementary propert of prefix scan
// If the predicate is true, we enqueu the value as before.
// If the predicate is false, we enqueu the value from the other side
extern "C" __global__ void EnqueueComplementKernel( u32 size, const int* input, int* output, int* counters )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	u32 laneIndex = threadIdx.x & ( warpSize - 1 );

	int val = 0;
	if( index < size ) val = input[index];

	// Predicate indicating the prefix scan input (binary)
	// Indicating whether we enqueue the input value or not
	bool enqueue = val & 1;

	// Binary warp-wise prefix scan
	int warpScan = ScanWarpBinary( enqueue );

	// Complemental prefix scan with (k=1)
	int complWarpScan = laneIndex - warpScan;

	// The last thread in the warp atomically adds it value to the global counter to get the offset
	// The value of the last thread is equal to the sum of all elements in the warp
	int warpOffset = 0;
	if( laneIndex == warpSize - 1 ) warpOffset = atomicAdd( &counters[0], warpScan + enqueue );

	// We exchange the offset of the warp using the shuffle instruction
	warpOffset = __shfl( warpOffset, warpSize - 1 );

	// Unfortunately, we have to use the second atomic add for the complement
	int complWarpOffset = 0;
	if( laneIndex == warpSize - 1 ) complWarpOffset = atomicAdd( &counters[1], complWarpScan + !enqueue );

	// Again, we exchange the offset for the complement
	complWarpOffset = __shfl( complWarpOffset, warpSize - 1 );

	// Elements satifying the predicate are enqueued from the front
	if( index < size && enqueue )
		output[warpOffset + warpScan] = val;
	// Elements not satifying the predicate are enqueued from the back
	else
		output[size - 1 - ( complWarpOffset + complWarpScan )] = val;
}
