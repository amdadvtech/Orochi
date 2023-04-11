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
	u64 ballot = __ballot( val != 0 );
	int pps = __popcll( ballot );
	int laneIndex = threadIdx.x & ( warpSize - 1 );
	return __popcll( ballot & ( ( 1ull << laneIndex ) - 1 ) );
}

__device__ int WriteOutputBinary( u32 size, const int* input, int* output, int* counter )
{
	const u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	const u32 laneIndex = threadIdx.x & ( warpSize - 1 );
	
	int val = 0;
	if( index < size ) val = input[index];

	int warpScan = ScanWarpBinary( val & 1 );
	int warpOffset;
	if( laneIndex == warpSize - 1 )
		warpOffset = atomicAdd( counter, warpScan + ( val & 1 ) );
	warpOffset = __shfl( warpOffset, warpSize - 1 );

	if( index < size && ( val & 1 ) )
		output[warpOffset + warpScan] = val;
}

__device__ int WriteOutput( u32 size, const int* input, int* output, int* counter )
{
	const u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	const u32 laneIndex = threadIdx.x & ( warpSize - 1 );

	int val = 0;
	if( index < size ) val = input[index];

	int warpScan = ScanWarp( val & 1 ) - ( val & 1 );
	int warpOffset;
	if( laneIndex == warpSize - 1 )
	{
		warpOffset = atomicAdd( counter, warpScan + ( val & 1 ) );
	}
	warpOffset = __shfl( warpOffset, warpSize - 1 );

	if( index < size && ( val & 1 ) )
		output[warpOffset + warpScan] = val;
}

extern "C" __global__ void WritingOutputKernel( u32 size, const int* input, int* output, int* counter )
{
	WriteOutput( size, input, output, counter );
}

extern "C" __global__ void WritingOutputBinaryKernel( u32 size, const int* input, int* output, int* counter )
{
	WriteOutputBinary( size, input, output, counter );
}
