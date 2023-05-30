
#include <09_RadixSort/Configs.h>
#include <common/Common.h>

extern "C" __global__ void CountKernel( int* gSrc, int* gDst, int gN, int gNItemsPerWG, const int START_BIT, const int N_WGS_EXECUTED )
{
	__shared__ int table[BIN_SIZE];

	for( int i = threadIdx.x; i < BIN_SIZE; i += COUNT_WG_SIZE )
	{
		table[i] = 0;
	}

	__syncthreads();

	const int offset = blockIdx.x * gNItemsPerWG;
	const int upperBound = ( offset + gNItemsPerWG > gN ) ? gN - offset : gNItemsPerWG;

	for( int i = threadIdx.x; i < upperBound; i += COUNT_WG_SIZE )
	{
		const int idx = offset + i;
		const int tableIdx = ( gSrc[idx] >> START_BIT ) & RADIX_MASK;
		atomicAdd( &table[tableIdx], 1 );
	}

	__syncthreads();

	// Assume COUNT_WG_SIZE == BIN_SIZE
	gDst[threadIdx.x * N_WGS_EXECUTED + blockIdx.x] = table[threadIdx.x];
}

template<typename T>
__device__ T ldsScanExclusive( T* lds, int width )
{
	__shared__ T sum;

	int offset = 1;

	for( int d = width >> 1; d > 0; d >>= 1 )
	{

		if( threadIdx.x < d )
		{
			const int firstInputIndex = offset * ( 2 * threadIdx.x + 1 ) - 1;
			const int secondInputIndex = offset * ( 2 * threadIdx.x + 2 ) - 1;

			lds[secondInputIndex] += lds[firstInputIndex];
		}
		__syncthreads();

		offset *= 2;
	}

	__syncthreads();

	if( threadIdx.x == 0 )
	{
		sum = lds[width - 1];
		__threadfence_block();

		lds[width - 1] = 0;
		__threadfence_block();
	}

	for( int d = 1; d < width; d *= 2 )
	{
		offset >>= 1;

		if( threadIdx.x < d )
		{
			const int firstInputIndex = offset * ( 2 * threadIdx.x + 1 ) - 1;
			const int secondInputIndex = offset * ( 2 * threadIdx.x + 2 ) - 1;

			const T t = lds[firstInputIndex];
			lds[firstInputIndex] = lds[secondInputIndex];
			lds[secondInputIndex] += t;
		}
		__syncthreads();
	}

	__syncthreads();

	return sum;
}

extern "C" __device__ void WorkgroupSync( int threadId, int blockId, int currentSegmentSum, int* currentGlobalOffset, volatile int* gPartialSum, volatile bool* gIsReady )
{
	if( threadId == 0 )
	{
		int offset = 0;

		if( blockId != 0 )
		{
			while( !gIsReady[blockId - 1] )
			{
			}

			offset = gPartialSum[blockId - 1];

			__threadfence();

			// Reset the value
			gIsReady[blockId - 1] = false;
		}

		gPartialSum[blockId] = offset + currentSegmentSum;

		// Ensure that the gIsReady is only modified after the gPartialSum is written.
		__threadfence();

		gIsReady[blockId] = true;

		*currentGlobalOffset = offset;
	}

	__syncthreads();
}

extern "C" __global__ void ParallelExclusiveScan( int* gCount, int* gHistogram, volatile int* gPartialSum, volatile bool* gIsReady )
{
	// Fill the LDS with the partial sum of each segment
	__shared__ int blockBuffer[SCAN_WG_SIZE];

	blockBuffer[threadIdx.x] = gCount[blockIdx.x * blockDim.x + threadIdx.x];

	__syncthreads();

	// Do parallel exclusive scan on the LDS

	int currentSegmentSum = ldsScanExclusive( blockBuffer, SCAN_WG_SIZE );

	__syncthreads();

	// Sync all the Workgroups to calculate the global offset.

	__shared__ int currentGlobalOffset;
	WorkgroupSync( threadIdx.x, blockIdx.x, currentSegmentSum, &currentGlobalOffset, gPartialSum, gIsReady );

	// Write back the result.

	gHistogram[blockIdx.x * blockDim.x + threadIdx.x] = blockBuffer[threadIdx.x] + currentGlobalOffset;
}

template<int N_ITEMS_PER_WI, int EXEC_WIDTH>
__device__ void localSort4bitMulti( int* keys, u32* ldsKeys, const int START_BIT )
{
	__shared__ union
	{
		u16 m_unpacked[EXEC_WIDTH + 1][N_BINS_PACKED_4BIT][N_BINS_PACK_FACTOR];
		u64 m_packed[EXEC_WIDTH + 1][N_BINS_PACKED_4BIT];
	} lds;

	__shared__ u64 ldsTemp[EXEC_WIDTH];

	for( int i = 0; i < N_BINS_PACKED_4BIT; ++i )
	{
		lds.m_packed[threadIdx.x][i] = 0UL;
	}

	for( int i = 0; i < N_ITEMS_PER_WI; ++i )
	{
		const int in4bit = ( keys[i] >> START_BIT ) & 0xf;
		const int packIdx = in4bit / N_BINS_PACK_FACTOR;
		const int idx = in4bit % N_BINS_PACK_FACTOR;
		lds.m_unpacked[threadIdx.x][packIdx][idx] += 1;
	}

	__syncthreads();

	for( int ii = 0; ii < N_BINS_PACKED_4BIT; ++ii )
	{
		ldsTemp[threadIdx.x] = lds.m_packed[threadIdx.x][ii];
		__syncthreads();
		const u64 sum = ldsScanExclusive( ldsTemp, EXEC_WIDTH );
		__syncthreads();
		lds.m_packed[threadIdx.x][ii] = ldsTemp[threadIdx.x];

		if( threadIdx.x == 0 ) lds.m_packed[EXEC_WIDTH][ii] = sum;
	}

	__syncthreads();

	auto* tmp = &lds.m_unpacked[EXEC_WIDTH][0][0];
	ldsScanExclusive( tmp, N_BINS_PACKED_4BIT * N_BINS_PACK_FACTOR );

	__syncthreads();

	for( int i = 0; i < N_ITEMS_PER_WI; ++i )
	{
		const int in4bit = ( keys[i] >> START_BIT ) & 0xf;
		const int packIdx = in4bit / N_BINS_PACK_FACTOR;
		const int idx = in4bit % N_BINS_PACK_FACTOR;
		const int offset = lds.m_unpacked[EXEC_WIDTH][packIdx][idx];
		const int rank = lds.m_unpacked[threadIdx.x][packIdx][idx]++;

		ldsKeys[offset + rank] = keys[i];
	}
	__syncthreads();

	for( int i = 0; i < N_ITEMS_PER_WI; ++i )
	{
		keys[i] = ldsKeys[threadIdx.x * N_ITEMS_PER_WI + i];
	}
}

__device__ void localSort8bitMulti( int* keys, u32* ldsKeys, const int START_BIT )
{
	localSort4bitMulti<SORT_N_ITEMS_PER_WI, SORT_WG_SIZE>( keys, ldsKeys, START_BIT );
	localSort4bitMulti<SORT_N_ITEMS_PER_WI, SORT_WG_SIZE>( keys, ldsKeys, START_BIT + 4 );
}

extern "C" __global__ void SortKernel( int* gSrcKey, int* gDstKey, int* gHistogram, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	int offset = blockIdx.x * blockDim.x * gNItemsPerWI;
	if( offset > gN )
	{
		return;
	}

	__shared__ u32 localOffsets[BIN_SIZE];
	__shared__ u32 ldsKeys[SORT_WG_SIZE * SORT_N_ITEMS_PER_WI];

	__shared__ union
	{
		u16 histogram[2][BIN_SIZE];
		u32 histogramU32[BIN_SIZE];
	} lds;

	int keys[SORT_N_ITEMS_PER_WI] = { 0 };

	for( int i = threadIdx.x; i < BIN_SIZE; i += SORT_WG_SIZE )
	{
		localOffsets[i] = gHistogram[i * N_WGS_EXECUTED + blockIdx.x];
	}
	__syncthreads();

	for( int ii = 0; ii < gNItemsPerWI; ii += SORT_N_ITEMS_PER_WI )
	{
		for( int i = 0; i < SORT_N_ITEMS_PER_WI; ++i )
		{
			const int idx = offset + i * SORT_WG_SIZE + threadIdx.x;
			ldsKeys[i * SORT_WG_SIZE + threadIdx.x] = ( idx < gN ) ? gSrcKey[idx] : 0xffffffff;
		}
		__syncthreads();

		for( int i = 0; i < SORT_N_ITEMS_PER_WI; ++i )
		{
			const int idx = threadIdx.x * SORT_N_ITEMS_PER_WI + i;
			keys[i] = ldsKeys[idx];
		}

		localSort8bitMulti( keys, ldsKeys, START_BIT );

		for( int i = threadIdx.x; i < BIN_SIZE; i += SORT_WG_SIZE )
		{
			lds.histogramU32[i] = 0;
		}
		__syncthreads();

		for( int i = 0; i < SORT_N_ITEMS_PER_WI; ++i )
		{
			const int a = threadIdx.x * SORT_N_ITEMS_PER_WI + i;
			const int b = a - 1;
			const int aa = ( ldsKeys[a] >> START_BIT ) & RADIX_MASK;
			const int bb = ( ( ( b >= 0 ) ? ldsKeys[b] : 0xffffffff ) >> START_BIT ) & RADIX_MASK;
			if( aa != bb )
			{
				lds.histogram[0][aa] = a;
				if( b >= 0 ) lds.histogram[1][bb] = a;
			}
		}
		if( threadIdx.x == 0 ) lds.histogram[1][( ldsKeys[SORT_N_ITEMS_PER_WI * SORT_WG_SIZE - 1] >> START_BIT ) & RADIX_MASK] = SORT_N_ITEMS_PER_WI * SORT_WG_SIZE;

		__syncthreads();

		const int upperBound = ( offset + threadIdx.x * SORT_N_ITEMS_PER_WI + SORT_N_ITEMS_PER_WI > gN ) ? ( gN - ( offset + threadIdx.x * SORT_N_ITEMS_PER_WI ) ) : SORT_N_ITEMS_PER_WI;
		if( upperBound < 0 )
		{
			return;
		}

		for( int i = 0; i < upperBound; ++i )
		{
			const int tableIdx = ( keys[i] >> START_BIT ) & RADIX_MASK;
			const int dstIdx = localOffsets[tableIdx] + ( threadIdx.x * SORT_N_ITEMS_PER_WI + i ) - lds.histogram[0][tableIdx];
			gDstKey[dstIdx] = keys[i];
		}

		__syncthreads();

		for( int i = 0; i < N_BINS_PER_WI; i++ )
		{
			const int idx = threadIdx.x * N_BINS_PER_WI + i;
			localOffsets[idx] += lds.histogram[1][idx] - lds.histogram[0][idx];
		}

		offset += SORT_WG_SIZE * SORT_N_ITEMS_PER_WI;
		if( offset > gN )
		{
			return;
		}
	}
}
