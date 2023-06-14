#include <common/Common.h>

template<typename T>
__device__ T ScanWarp( T val )
{
	int laneIndex = threadIdx.x & ( warpSize - 1 );
	for( int i = 1; i < warpSize; i <<= 1 )
	{
		T tmp = __shfl_up( val, i );
		if( laneIndex >= i ) val += tmp;
	}
	return val;
}

__device__ int ScanWarpBinary( bool val )
{
	int laneIndex = threadIdx.x & ( warpSize - 1 );
	u64 ballot = __ballot( val );
	return __popcll( ballot & ( ( 1ull << laneIndex ) - 1ull ) );
}

extern "C" __global__ void EnqueueNaiveKernel( u32 size, const int* input, int* output, int* counter )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;

	int val = 0;
	if( index < size ) val = input[index];

	bool predicate = val & 1;
	if( index < size && predicate ) output[atomicAdd( counter, 1 )] = val;
}

extern "C" __global__ void EnqueueKernel( u32 size, const int* input, int* output, int* counter )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	u32 laneIndex = threadIdx.x & ( warpSize - 1 );

	int val = 0;
	if( index < size ) val = input[index];

	bool predicate = val & 1;
	int warpScan = ScanWarp( predicate ? 1 : 0 ) - predicate;

	int warpOffset = 0;
	if( laneIndex == warpSize - 1 ) warpOffset = atomicAdd( counter, warpScan + predicate );
	warpOffset = __shfl( warpOffset, warpSize - 1 );

	if( index < size && predicate ) output[warpOffset + warpScan] = val;
}

extern "C" __global__ void EnqueueBinaryKernel( u32 size, const int* input, int* output, int* counter )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	u32 laneIndex = threadIdx.x & ( warpSize - 1 );

	int val = 0;
	if( index < size ) val = input[index];

	bool predicate = val & 1;
	int warpScan = ScanWarpBinary( predicate );

	int warpOffset = 0;
	if( laneIndex == warpSize - 1 ) warpOffset = atomicAdd( counter, warpScan + predicate );
	warpOffset = __shfl( warpOffset, warpSize - 1 );

	if( index < size && predicate ) output[warpOffset + warpScan] = val;
}

extern "C" __global__ void EnqueueComplementKernel( u32 size, const int* input, int* output, int* counters )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	u32 laneIndex = threadIdx.x & ( warpSize - 1 );

	int val = 0;
	if( index < size ) val = input[index];

	bool predicate = val & 1;
	int warpScan = ScanWarpBinary( predicate );
	int complWarpScan = laneIndex - warpScan;

	int warpOffset = 0;
	if( laneIndex == warpSize - 1 ) warpOffset = atomicAdd( &counters[0], warpScan + predicate );
	warpOffset = __shfl( warpOffset, warpSize - 1 );

	int complWarpOffset = 0;
	if( laneIndex == warpSize - 1 ) complWarpOffset = atomicAdd( &counters[1], complWarpScan + !predicate );
	complWarpOffset = __shfl( complWarpOffset, warpSize - 1 );

	if( index < size && predicate )
		output[warpOffset + warpScan] = val;
	else
		output[size - 1 - ( complWarpOffset + complWarpScan )] = val;
}
