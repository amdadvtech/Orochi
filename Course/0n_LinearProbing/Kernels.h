#include <common/Common.h>
#include <LP.h>

extern "C" __global__ void insert( LP_Concurrent<false> lp, int upper, int nItemsPerThread )
{ 
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	splitmix64 rnd;
	rnd.x = tid;

	for( int i = 0; i < nItemsPerThread; i++ )
	{
		int x = rnd.next() % upper;
		lp.insert( x );
	}
}