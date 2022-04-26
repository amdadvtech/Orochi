#include <Test/ParallelPrimitives/RadixSortConfigs.h>
#define LDS_BARRIER __syncthreads()

using namespace Oro;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

// #define NV_WORKAROUND 1

#define THE_FIRST_THREAD threadIdx.x == 0 && blockIdx.x == 0

extern "C"
__global__ void CountKernelReference( int* gSrc, int* gDst, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;

	const int offset = blockIdx.x * blockDim.x * gNItemsPerWI;

	int table[BIN_SIZE] = {0};

	for( int i = 0; i < gNItemsPerWI; i++)
	{
		int idx = offset + threadIdx.x * gNItemsPerWI + i;

		if( idx >= gN )
			continue;
		int tableIdx = ( gSrc[idx] >> START_BIT ) & RADIX_MASK;
		table[tableIdx] ++;
	}
	
	const int wgIdx = blockIdx.x;

	for(int i=0; i<BIN_SIZE; i++)
	{
		if( table[i] != 0 )
		{
			atomicAdd( &gDst[i * N_WGS_EXECUTED + wgIdx], table[i] );
		}
	}
}

extern "C" 
__global__ void SortKernelReference( int* gSrc, int* gDst, int* gHistogram, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int offset = blockIdx.x * blockDim.x * gNItemsPerWI;
	const int wgIdx = blockIdx.x;

	__shared__ int lds[WG_SIZE];
	__shared__ int localHistogram[BIN_SIZE];
	__shared__ int localOffsets[BIN_SIZE];

	for(int i=threadIdx.x; i<BIN_SIZE; i+=WG_SIZE)
	{
		localOffsets[i] = gHistogram[i * N_WGS_EXECUTED + wgIdx ];
	}
	LDS_BARRIER;

	for( int i = 0; i < gNItemsPerWI; i++ )
	{
		int idx = offset + threadIdx.x * gNItemsPerWI + i;

		int key;
		int tableIdx;
		if( idx < gN )
		{
			key = gSrc[idx];
			tableIdx = ( key >> START_BIT ) & RADIX_MASK;
		}

		LDS_BARRIER;
		if( idx < gN )
		{
			int dstIdx = atomicAdd( &localOffsets[tableIdx], 1 );
			gDst[dstIdx] = key;
		}
	}
}


//=====
extern "C" __global__ void CountKernel( int* gSrc, int* gDst, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int offset = blockIdx.x * blockDim.x * gNItemsPerWI;

	__shared__ union
	{
		u8 m_wiCounter[WG_SIZE][N_PACKED][PACK_FACTOR];//can share a counter among WIs to reduce lds
		int m_wiPackedCounter[WG_SIZE][N_PACKED];
	} lds;

	int table[N_PACKED_PER_WI][PACK_FACTOR] = { 0 };

	for( int iter = 0; iter < gNItemsPerWI; iter+=255 )
	{
		LDS_BARRIER;
		for(int i=0; i<N_PACKED; i++)
			lds.m_wiPackedCounter[threadIdx.x][i] = 0;

		for(int i=0; i<min(255, gNItemsPerWI-iter); i++)
		{
			int idx = offset + (iter+i) * WG_SIZE + threadIdx.x;
			if( idx < gN )
			{
				int tableIdx = ( gSrc[idx] >> START_BIT ) & RADIX_MASK;
				int s = tableIdx / PACK_FACTOR;
				int t = tableIdx % PACK_FACTOR;
				lds.m_wiCounter[threadIdx.x][s][t]++;
			}
		}

		LDS_BARRIER;
		// put them back to vgpr
		for(int i=0; i<N_PACKED_PER_WI; i++)
		{
			for( int k = 0; k < PACK_FACTOR; k++ )
			{
				int sum = 0;
				int ii = threadIdx.x * N_PACKED_PER_WI + i;
				for( int j = 0; j < WG_SIZE; j++ )
				{
					sum += lds.m_wiCounter[j][ii][k];
				}
				table[i][k] += sum;
			}
		}
	}

	const int wgIdx = blockIdx.x;

	for( int i = 0; i < N_PACKED_PER_WI; i++ )
	{
		int ii = threadIdx.x * N_PACKED_PER_WI + i;
		for( int k = 0; k < PACK_FACTOR; k++ )
		{
			int binIdx = ii * PACK_FACTOR + k;
			gDst[binIdx * N_WGS_EXECUTED + wgIdx] = table[i][k];
		}
	}
}

__device__ int ldsScan( int* lds, int width ) 
{
	int idx = threadIdx.x;
	for( int i = 1; i < width; i*=2 )
	{
		if( idx >= i ) lds[idx] += lds[idx - i];
		LDS_BARRIER;
	}
/*
	if( idx >= 1 ) lds[idx] += lds[idx - 1];
	LDS_BARRIER
	if( idx >= 2 ) lds[idx] += lds[idx - 2];
	LDS_BARRIER
	if( idx >= 4 ) lds[idx] += lds[idx - 4];
	LDS_BARRIER
	if( idx >= 8 ) lds[idx] += lds[idx - 8];
	LDS_BARRIER
	if( idx >= 16 ) lds[idx] += lds[idx - 16];
	LDS_BARRIER
*/
	//	if( idx >= 32 ) lds[idx] += lds[idx - 32];

	LDS_BARRIER;
	int sum = lds[width-1];
	LDS_BARRIER;

	int t = (idx==0)?0:lds[idx-1];
	lds[idx] = t;

	LDS_BARRIER;
	return sum;
}

template<typename T, int STRIDE>
struct ScanImpl
{
	__device__ static T exec( T a )
	{
		T b = __shfl( a, threadIdx.x - STRIDE );
		if( threadIdx.x >= STRIDE ) a += b;
		return ScanImpl<T,STRIDE * 2>::exec( a );
	}
};

template<typename T>
struct ScanImpl<T,WG_SIZE>
{
	__device__ static T exec( T a )
	{
		return a;
	}
};

template<typename T>
__device__ 
void waveScanInclusive( T& a, int width ) 
{
#if 0
	a = ScanImpl<T, 1>::exec( a );
#else
	for( int i = 1; i < width; i *= 2 )
	{
		T b = __shfl( a, threadIdx.x - i );
		if( threadIdx.x >= i ) a += b;
	}
#endif
}

template<typename T>
__device__ T waveScanExclusive( T& a, int width )
{
	waveScanInclusive( a, width );

	T sum = __shfl( a, width-1 );
	a = __shfl( a, threadIdx.x-1 );
	if( threadIdx.x == 0 ) a = 0;

	return sum;
}

template<typename T>
__device__ void ldsScanInclusive( T* lds, int width )
{
	// The width cannot exceed WG_SIZE
	__shared__ T temp[2][WG_SIZE];

	constexpr int MAX_INDEX = 1;
	int outIndex = 0;
	int inIndex = 1;

	temp[outIndex][threadIdx.x] = lds[threadIdx.x];
	LDS_BARRIER;

	for( int i = 1; i < width; i *= 2 )
	{
		// Swap in and out index for the buffers

		outIndex = MAX_INDEX - outIndex;
		inIndex = MAX_INDEX - outIndex;

		if( threadIdx.x >= i )
		{
			temp[outIndex][threadIdx.x] = temp[inIndex][threadIdx.x] + temp[inIndex][threadIdx.x - i];
		}
		else
		{
			temp[outIndex][threadIdx.x] = temp[inIndex][threadIdx.x];
		}

		LDS_BARRIER;
	}

	lds[threadIdx.x] = temp[outIndex][threadIdx.x];

	// Ensure the results are written in LDS and are observable in a block (workgroup) before return.
	__threadfence_block();
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
		LDS_BARRIER;

		offset *= 2;
	}

	LDS_BARRIER;

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
		LDS_BARRIER;
	}

	LDS_BARRIER;

	return sum;
}
//========================

__device__ void localSort4bitMultiRef( int* keys, u32* ldsKeys, const int START_BIT )
{
	__shared__ u32 ldsTemp[WG_SIZE + 1][N_BINS_4BIT];

	for( int i = 0; i < N_BINS_4BIT; i++ )
	{
		ldsTemp[threadIdx.x][i] = 0;
	}
	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		int in4bit = ( keys[i] >> START_BIT ) & 0xf;
		ldsTemp[threadIdx.x][in4bit] += 1;
	}

	LDS_BARRIER;

	if( threadIdx.x < N_BINS_4BIT )//16 scans, pack 4 scans into 1 to make 4 parallel scans
	{
		int sum = 0;
		for( int i = 0; i < WG_SIZE; i++ )
		{
			int t = ldsTemp[i][threadIdx.x];
			ldsTemp[i][threadIdx.x] = sum;
			sum += t;
		}
		ldsTemp[WG_SIZE][threadIdx.x] = sum;
	}
	LDS_BARRIER;
	if( threadIdx.x == 0 )//todo parallel scan
	{
		int sum = 0;
		for( int i = 0; i < N_BINS_4BIT; i++ )
		{
			int t = ldsTemp[WG_SIZE][i];
			ldsTemp[WG_SIZE][i] = sum;
			sum += t;
		}
	}
	LDS_BARRIER;

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		int in4bit = ( keys[i] >> START_BIT ) & 0xf;
		int offset = ldsTemp[WG_SIZE][in4bit];
		int rank = ldsTemp[threadIdx.x][in4bit]++;

		ldsKeys[offset + rank] = keys[i];
	}
	LDS_BARRIER;

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		keys[i] = ldsKeys[threadIdx.x * SORT_N_ITEMS_PER_WI + i];
	}
}

__device__ void localSort4bitMulti( int* keys, u32* ldsKeys, const int START_BIT )
{
	__shared__ union
	{
		u16 m_unpacked[WG_SIZE + 1][N_BINS_PACKED_4BIT][N_BINS_PACK_FACTOR];
		u64 m_packed[WG_SIZE + 1][N_BINS_PACKED_4BIT];
	} lds;
	__shared__ u64 ldsTemp[WG_SIZE];//todo. remove me

	for( int i = 0; i < N_BINS_PACKED_4BIT; i++ )
	{
		lds.m_packed[threadIdx.x][i] = 0;
	}

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		int in4bit = ( keys[i] >> START_BIT ) & 0xf;
		int packIdx = in4bit/N_BINS_PACK_FACTOR;
		int idx = in4bit % N_BINS_PACK_FACTOR;
		lds.m_unpacked[threadIdx.x][packIdx][idx] += 1;
	}

	LDS_BARRIER;

#if defined( NV_WORKAROUND )
	if( threadIdx.x < N_BINS_PACKED_4BIT ) // 16 scans, pack 4 scans into 1 to make 4 parallel scans
	{
		u64 sum = 0;
		for( int i = 0; i < WG_SIZE; i++ )
		{
			u64 t = lds.m_packed[i][threadIdx.x];
			lds.m_packed[i][threadIdx.x] = sum;
			sum += t;
		}
		lds.m_packed[WG_SIZE][threadIdx.x] = sum;
	}
#else
	for( int ii = 0; ii < N_BINS_PACKED_4BIT; ii++)
	{
		ldsTemp[threadIdx.x] = lds.m_packed[threadIdx.x][ii];
		LDS_BARRIER;
		u64 sum = ldsScanExclusive( ldsTemp, WG_SIZE );
		LDS_BARRIER;
		lds.m_packed[threadIdx.x][ii] = ldsTemp[threadIdx.x];

		if( threadIdx.x == 0 ) lds.m_packed[WG_SIZE][ii] = sum;
	}
#endif
	LDS_BARRIER;
	if( threadIdx.x == 0 ) // todo. parallel scan
	{
		int sum = 0;
		for( int i = 0; i < N_BINS_PACKED_4BIT; i++ )
		{
			for( int j = 0; j < N_BINS_PACK_FACTOR; j++)
			{
				int t = lds.m_unpacked[WG_SIZE][i][j];
				lds.m_unpacked[WG_SIZE][i][j] = sum;
				sum += t;
			}
		}
	}
	LDS_BARRIER;

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		int in4bit = ( keys[i] >> START_BIT ) & 0xf;
		int packIdx = in4bit / N_BINS_PACK_FACTOR;
		int idx = in4bit % N_BINS_PACK_FACTOR;
		int offset = lds.m_unpacked[WG_SIZE][packIdx][idx];
		int rank = lds.m_unpacked[threadIdx.x][packIdx][idx]++;

		ldsKeys[offset + rank] = keys[i];
	}
	LDS_BARRIER;

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		keys[i] = ldsKeys[threadIdx.x * SORT_N_ITEMS_PER_WI + i];
	}
}

__device__ void localSort8bitMulti( int* keys, u32* ldsKeys, const int START_BIT )
{
	localSort4bitMulti( keys, ldsKeys, START_BIT );
	if( N_RADIX > 4 ) localSort4bitMulti( keys, ldsKeys, START_BIT + 4 );
}

extern "C" __global__ void SortKernel( int* gSrc, int* gDst, int* gHistogram, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockIdx.x * blockDim.x * gNItemsPerWI;
	const int wgIdx = blockIdx.x;

	__shared__ u32 localOffsets[BIN_SIZE];

	__shared__ u32 ldsKeys[WG_SIZE * SORT_N_ITEMS_PER_WI]; // todo. can be aliased

	__shared__ int ldsHistogram[BIN_SIZE]; // todo. can be aliased
#if 0
	{
		int a = 1;

		int width = WG_SIZE;
/*
		for( int i = 1; i < width; i*=2 )
		{
			int b = __shfl( a, threadIdx.x - i );
			if( threadIdx.x >= i ) a += b;
		}
*/
		a = 1;
		ldsHistogram[threadIdx.x] = a;
		LDS_BARRIER;
		int sum = ldsScanExclusive( ldsHistogram, width );
		LDS_BARRIER;
		if( THE_FIRST_THREAD )
		{
			for(int i=0; i<WG_SIZE; i++)
				printf( "%d,", ldsHistogram[i] );
			printf( "\n" );
			printf("%d\n", sum);
		}
	}
#endif
	int histogram[N_BINS_PER_WI] = { 0 };
	int keys[SORT_N_ITEMS_PER_WI] = { 0 };

	for( int i = threadIdx.x; i < BIN_SIZE; i += WG_SIZE )
	{
		localOffsets[i] = gHistogram[i * N_WGS_EXECUTED + wgIdx];
	}
	LDS_BARRIER;

	for( int ii = 0; ii < gNItemsPerWI; ii += SORT_N_ITEMS_PER_WI )
	{
		for(int i=0; i<SORT_N_ITEMS_PER_WI; i++)
		{
			int idx = offset + threadIdx.x * SORT_N_ITEMS_PER_WI + i;
			keys[i] = (idx<gN)? gSrc[idx] : 0xffffffff;
		}

		//local sort keys[];
		localSort8bitMulti( keys, ldsKeys, START_BIT );
#if 0
		if( THE_FIRST_THREAD )
		{
			for( int i = 0; i < WG_SIZE * SORT_N_ITEMS_PER_WI ; i++)
				printf("%d,", ldsKeys[i]);
			printf("\n");
		}
		break;
#endif	
		for( int i = threadIdx.x; i < BIN_SIZE; i += WG_SIZE )
			ldsHistogram[i] = 0;
		LDS_BARRIER;
		for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
		{
			int tableIdx = ( keys[i] >> START_BIT ) & RADIX_MASK;
			atomicAdd( &ldsHistogram[tableIdx], 1 );
		}
		LDS_BARRIER;
		for( int i = 0; i < N_BINS_PER_WI; i++ )
		{
			histogram[i] = ldsHistogram[threadIdx.x * N_BINS_PER_WI + i];
		}
#if defined( NV_WORKAROUND )
		if( threadIdx.x == 0 ) // todo. parallel scan
		{
			int sum = 0;
			for( int i = 0; i < BIN_SIZE; i++ )
			{
				int t = ldsHistogram[i];
				ldsHistogram[i] = sum;
				sum += t;
			}
		}
#else
		int sum = 0;
		for( int i = 0; i < BIN_SIZE; i+=WG_SIZE)
		{
			int* dst = ldsHistogram + i;
			int t = ldsScanExclusive( dst, WG_SIZE );
			dst[threadIdx.x] += sum;
			sum += t;
		}
#endif
		LDS_BARRIER;
		for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
		{
			int idx = offset + threadIdx.x * SORT_N_ITEMS_PER_WI + i;
			if( idx < gN )
			{
				int tableIdx = ( keys[i] >> START_BIT ) & RADIX_MASK;
				int dstIdx = localOffsets[tableIdx] + (threadIdx.x*SORT_N_ITEMS_PER_WI+i) - ldsHistogram[tableIdx];
				gDst[dstIdx] = keys[i];
			}
		}
		LDS_BARRIER;

		for( int i = 0; i < N_BINS_PER_WI; i++ )
		{
			int idx = threadIdx.x * N_BINS_PER_WI + i;
			localOffsets[idx] += histogram[i];
		}
		//===
		offset += WG_SIZE * SORT_N_ITEMS_PER_WI;
	}
}

extern "C" __global__ void SortKernel1( int* gSrc, int* gDst, int* gHistogram, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockIdx.x * blockDim.x * gNItemsPerWI;
	const int wgIdx = blockIdx.x;

	__shared__ u32 localOffsets[BIN_SIZE];

	__shared__ u32 ldsKeys[WG_SIZE * SORT_N_ITEMS_PER_WI]; // todo. can be aliased

	__shared__ union
	{
		u16 histogram[2][BIN_SIZE]; // low and high// todo. can be aliased
		u32 histogramU32[BIN_SIZE];
	} lds;

	int keys[SORT_N_ITEMS_PER_WI] = { 0 };

	for( int i = threadIdx.x; i < BIN_SIZE; i += WG_SIZE )
	{
		localOffsets[i] = gHistogram[i * N_WGS_EXECUTED + wgIdx];
	}
	LDS_BARRIER;

	for( int ii = 0; ii < gNItemsPerWI; ii += SORT_N_ITEMS_PER_WI )
	{
		for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
		{
			int idx = offset + threadIdx.x * SORT_N_ITEMS_PER_WI + i;
			keys[i] = ( idx < gN ) ? gSrc[idx] : 0xffffffff;
		}

		// local sort keys[];
		localSort8bitMulti( keys, ldsKeys, START_BIT );
#if 0
		if( THE_FIRST_THREAD )
		{
			for( int i = 0; i < WG_SIZE * SORT_N_ITEMS_PER_WI ; i++)
				printf("%d,", ldsKeys[i]);
			printf("\n");
		}
		break;
#endif
		for( int i = threadIdx.x; i < BIN_SIZE; i += WG_SIZE )
			lds.histogramU32[i] = 0;
		LDS_BARRIER;

		for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
		{
			int a = threadIdx.x * SORT_N_ITEMS_PER_WI + i;
			int b = a - 1;
			int aa = ( ldsKeys[a] >> START_BIT ) & RADIX_MASK;
			int bb = ( ( ( b >= 0 ) ? ldsKeys[b] : 0xffffffff ) >> START_BIT ) & RADIX_MASK;
			if( aa != bb )
			{
				lds.histogram[0][aa] = a;
				if( b >= 0 ) lds.histogram[1][bb] = a;
			}
		}
		if( threadIdx.x == 0 ) lds.histogram[1][( ldsKeys[SORT_N_ITEMS_PER_WI * WG_SIZE - 1] >> START_BIT ) & RADIX_MASK] = SORT_N_ITEMS_PER_WI * WG_SIZE;

		LDS_BARRIER;

		for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
		{
			int idx = offset + threadIdx.x * SORT_N_ITEMS_PER_WI + i;
			if( idx < gN )
			{
				int tableIdx = ( keys[i] >> START_BIT ) & RADIX_MASK;
				int dstIdx = localOffsets[tableIdx] + ( threadIdx.x * SORT_N_ITEMS_PER_WI + i ) - lds.histogram[0][tableIdx];
				gDst[dstIdx] = keys[i];
			}
		}
		LDS_BARRIER;

		for( int i = 0; i < N_BINS_PER_WI; i++ )
		{
			int idx = threadIdx.x * N_BINS_PER_WI + i;
			localOffsets[idx] += lds.histogram[1][idx] - lds.histogram[0][idx];
		}
		//===
		offset += WG_SIZE * SORT_N_ITEMS_PER_WI;
	}
}


extern "C" __global__ void ParallelExclusiveScanSingleWG( int* gCount, int* gHistogram, const int N_WGS_EXECUTED )
{
	// Use a single WG.
	if( blockIdx.x != 0 )
	{
		return;
	}

	// LDS for the parallel scan of the global sum:
	// First we store the sum of the counters of each number to it,
	// then we compute the global offset using parallel exclusive scan.
	__shared__ int blockBuffer[BIN_SIZE];

	// fill the LDS with the local sum

	for( int binId = threadIdx.x; binId < BIN_SIZE; binId += WG_SIZE )
	{
		// Do exclusive scan for each segment handled by each WI in a WG

		int localThreadSum = 0;
		for( int i = 0; i < N_WGS_EXECUTED; ++i )
		{
			int current = gCount[binId * N_WGS_EXECUTED + i];
			gCount[binId * N_WGS_EXECUTED + i] = localThreadSum;

			localThreadSum += current;
		}

		// Store the thread local sum to LDS.

		blockBuffer[binId] = localThreadSum;
	}

	LDS_BARRIER;

	// Do parallel exclusive scan on the LDS

	int globalSum = 0;
	for( int binId = 0; binId < BIN_SIZE; binId += WG_SIZE * 2 )
	{
		int* globalOffset = &blockBuffer[binId];
		int currentGlobalSum = ldsScanExclusive( globalOffset, WG_SIZE * 2 );
		globalOffset[threadIdx.x * 2] += globalSum;
		globalOffset[threadIdx.x * 2 + 1] += globalSum;
		globalSum += currentGlobalSum;
	}

	LDS_BARRIER;

	// Add the global offset to the global histogram.

	for( int binId = threadIdx.x; binId < BIN_SIZE; binId += WG_SIZE )
	{
		for( int i = 0; i < N_WGS_EXECUTED; ++i )
		{
			gHistogram[binId * N_WGS_EXECUTED + i] += blockBuffer[binId];
		}
	}
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

	LDS_BARRIER;
}

extern "C" __global__ void ParallelExclusiveScanAllWG( int* gCount, int* gHistogram, volatile int* gPartialSum, volatile bool* gIsReady )
{
	// Fill the LDS with the partial sum of each segment
	__shared__ int blockBuffer[SCAN_WG_SIZE];

	blockBuffer[threadIdx.x] = gCount[blockIdx.x * blockDim.x + threadIdx.x];

	LDS_BARRIER;

	// Do parallel exclusive scan on the LDS

	int currentSegmentSum = ldsScanExclusive( blockBuffer, SCAN_WG_SIZE );

	LDS_BARRIER;

	// Sync all the Workgroups to calculate the global offset.

	__shared__ int currentGlobalOffset;
	WorkgroupSync( threadIdx.x, blockIdx.x, currentSegmentSum, &currentGlobalOffset, gPartialSum, gIsReady );

	// Write back the result.

	gHistogram[blockIdx.x * blockDim.x + threadIdx.x] = blockBuffer[threadIdx.x] + currentGlobalOffset;
}
