#include <common/Common.h>
#include <LP.h>

extern "C" __global__ void insertLP( LP_Concurrent<false> lp, int upper, int nItemsPerThread )
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
extern "C" __global__ void insertBLP( BLP_ConcurrentGPU blp, int upper, int nItemsPerThread )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	splitmix64 rnd;
	rnd.x = tid;

	for( int i = 0; i < nItemsPerThread; i++ )
	{
		int x = rnd.next() % upper;
		blp.insert( x );
	}
}

extern "C" __global__ void findLP( LP_Concurrent<false> lp, int upper, int nItemsPerThread, u32* counter )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Find the same sequence
	{
		splitmix64 rnd;
		rnd.x = tid;

		int found = 0;
		for( int i = 0; i < nItemsPerThread; i++ )
		{
			int x = rnd.next() % upper;
			if( lp.find( x ) )
			{
				found++;
			}
		}
		atomicAdd( counter, found );
	}

	// Find another random case
	{
		splitmix64 rnd;
		rnd.x = tid ^ 0x12345;

		int found = 0;
		for( int i = 0; i < nItemsPerThread; i++ )
		{
			int x = rnd.next() % upper;
			if( lp.find( x ) )
			{
				found++;
			}
		}
		atomicAdd( counter, found );
	}
}

extern "C" __global__ void findBLP( BLP_ConcurrentGPU lp, int upper, int nItemsPerThread, u32* counter )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Find the same sequence
	{
		splitmix64 rnd;
		rnd.x = tid;

		int found = 0;
		for( int i = 0; i < nItemsPerThread; i++ )
		{
			int x = rnd.next() % upper;
			if( lp.find( x ) )
			{
				found++;
			}
		}
		atomicAdd( counter, found );
	}

	// Find another random case
	{
		splitmix64 rnd;
		rnd.x = tid ^ 0x12345;

		int found = 0;
		for( int i = 0; i < nItemsPerThread; i++ )
		{
			int x = rnd.next() % upper;
			if( lp.find( x ) )
			{
				found++;
			}
		}
		atomicAdd( counter, found );
	}
}

#if 0
// Dead lock example. Be careful if you execute without thread independent scheduling.
extern "C" __global__ void increment( u32 * counter, u32* mutex )
{
	while( atomicCAS( mutex, 0, 1 ) != 0 )
		;

	__threadfence();
	
	( *counter )++;

	__threadfence();
	atomicExch( mutex, 0 );
}
#else
// Exclusive lock example without dead locks.
extern "C" __global__ void increment( u32* counter, u32* mutex )
{
	// workaround
	u32 done = 0;
	do
	{
		if( done == 0 && atomicCAS( mutex, 0, 1 ) == 0 )
		{
			__threadfence();

			( *counter )++;

			__threadfence();
			atomicExch( mutex, 0 );

			done = 1;
		}
#if defined( CUDART_VERSION ) && CUDART_VERSION >= 9000
		__syncwarp();
	} while( __all_sync( 0xFFFFFFFF, done ) == false );
#else
	} while( __all( done ) == false );
#endif
}
#endif