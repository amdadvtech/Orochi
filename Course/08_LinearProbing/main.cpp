#include <common/Common.h>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <mutex>
#include <Windows.h>
#include <ppl.h>
#include <08_LinearProbing/LP.h>
#include <08_LinearProbing/LP_CPU.h>

template <class T>
void runTest( )
{
	int NBuckets = 1000;
	int Numbers = 10000;
	splitmix64 rnd;

	for( int i = 0; i < 10000; i++ )
	{
		T lp( NBuckets );
		std::set<uint32_t> s;
		for( int j = 0; j < NBuckets * 0.75; j++ )
		{
			uint32_t v = rnd.next() % Numbers;
			bool inserted = s.insert( v ).second;
			auto result = lp.insert( v );

			OROASSERT( result == ( inserted ? T::INSERTED : T::FOUND ), 0 );
		}
		OROASSERT( s == lp.set(), 0 );

		for( int i = 0; i < NBuckets * 0.75; i++ )
		{
			uint32_t v = rnd.next() % Numbers;
			bool found0 = s.count( v ) != 0;
			bool found1 = lp.find( v );

			OROASSERT( found0 == found1, 0 );
		}
	}
	printf( "Sequential Test: %s OK\n", typeid( T ).name() );
}

template <class T>
void runConcurrentTest( )
{ 
	int NThreads = 32;
	int NBuckets  = 10000;
	int Numbers = 1000000;
	double loadFactor = 0.75;

	for (int k = 0; k < 100; k++ )
	{
		StdSet_Concurrent truth;
		T storage( NBuckets );

		for( int i = 0; i < NThreads; i++ )
		{
			int nItemPerThread = NBuckets * loadFactor / NThreads;
			concurrency::parallel_for( 0, NThreads, [k, nItemPerThread, Numbers, &truth, &storage](int index) 
			{
				splitmix64 rnd;
				rnd.x = k * 1000000 + index;
				for( int j = 0; j < nItemPerThread; j++ )
				{
					uint32_t v = rnd.next() % Numbers;
					storage.insert( v );
					truth.insert( v );
				}
			} );
		}
		OROASSERT( truth.set() == storage.set(), 0 );

		splitmix64 rnd;
		for( int i = 0; i < NBuckets; i++ )
		{
			uint32_t v = rnd.next() % Numbers;
			OROASSERT( storage.find( v ) == truth.find( v ), 0 );
		}
	}
	printf( "Concurrent Test: %s OK\n", typeid( T ).name() );
}

inline int div_round_up( int val, int divisor ) 
{
	return ( val + divisor - 1 ) / divisor;
}

class LPSample : public Sample
{
public:
	LPSample() : Sample()
	{
		oroStreamCreate( &m_stream );
	}
	~LPSample() { 
		oroStreamDestroy( m_stream ); 
	}
	void launch1D( const char* function, unsigned int gridDimX, unsigned int blockDimX, std::initializer_list<void*> args )
	{
		std::vector<const char*> opts = {
			"-I../",
			"-I../08_LinearProbing/",
		};
		oroFunction f = m_utils.getFunctionFromFile( m_device, "../08_LinearProbing/Kernels.h", function, &opts );
		oroModuleLaunchKernel( f, gridDimX, 1, 1, blockDimX, 1, 1, 0, m_stream, std::vector<void*>( args ).data(), 0 );
	}
	oroStream getStream() { return m_stream; }
	oroStream m_stream = 0;
};



int main( int argc, char** argv )
{
	printf( "--- Test on the CPU ---\n" );
	// Test the correctness of Linear Probing and Bidirectional Linear Probing
	runTest<LP>();
	runTest<BLP>();
	runTest<LP_ConcurrentCPU>();
	runTest<BLP_ConcurrentCPU>();
	runConcurrentTest<LP_ConcurrentCPU>();
	runConcurrentTest<BLP_ConcurrentCPU>();


	printf( "--- Test on the GPU ---\n" );
	LPSample sample;

	int BlockSize = 32;
	int NBuckets = 100000000;
	int upper    = 1000000000;

// GPU varidation code. It takes long time.
//#define ENABLE_VARIDATION_GPU 1

#if defined( ENABLE_VARIDATION_GPU )
	std::set<uint32_t> referenceSet;
	for( int tid = 0; tid < nBlocks * BlockSize; tid++ )
	{
		splitmix64 rnd;
		rnd.x = tid;

		for( int i = 0; i < nItemsPerThread; i++ )
		{
			int x = rnd.next() % upper;
			referenceSet.insert( x );
		}
	}
#endif

	// load factor list for the benchmark 
	const double loadFactors[] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
	const int NLoadFactors = sizeof( loadFactors ) / sizeof( loadFactors[0] );
#if 1
	uint64_t itemsToIssue[NLoadFactors] =
	{ 
		10050080,
		20202421,
		30459635,
		40821942,
		51293228,
		61873878,
		72569211,
		83380440,
		94310970,
	};
#else
	uint64_t itemsToIssue[NLoadFactors];
	for( int i = 0; i < NLoadFactors; i++ )
	{
		LP lp( NBuckets );

		splitmix64 rnd;
		rnd.x = 0;

		int items = 0;
		int nIterations = 0;
		while( items < NBuckets * loadFactors[i] )
		{
			nIterations++;
			int x = rnd.next() % upper;

			if( lp.insert( x ) == LP::INSERTED )
			{
				items++;
			}
		}
		itemsToIssue[i] = nIterations;
		printf( "loadFactor %f : nIterations %d\n", loadFactors[i], nIterations );
	}
#endif

	// Linear Probing execution
	// 1. "insertLP" kernel - insert random numbers
	// 2. "findLP" kernel - find inserted numbers and find random numbers
	for( int i = 0; i < NLoadFactors; i++ )
	{
		int nItemsPerThread = 512;

		double insertionRatio = (double)( upper - NBuckets ) / upper;
		int nBlocks = div_round_up( itemsToIssue[i] / nItemsPerThread, BlockSize );

		int NRuns = 4;
		double sumExecMSInsert = 0.0;
		double sumExecMSFind = 0.0;
		for( int i = 0; i < NRuns + 1; i++ )
		{
			LP_ConcurrentGPU lpGpu( NBuckets );

			{
				OroStopwatch oroStream( sample.getStream() );
				oroStream.start();
				sample.launch1D( "insertLP", nBlocks, BlockSize, { &lpGpu, &upper, &nItemsPerThread } );
				oroStream.stop();
				float ms = oroStream.getMs();

				if( 0 < i )
					sumExecMSInsert += ms;
			}

			{
				BufferGPU<u32> counter;
				counter.resize( 1 );
				counter.fillZero();
				OroStopwatch oroStream( sample.getStream() );
				oroStream.start();
				sample.launch1D( "findLP", nBlocks, BlockSize, { &lpGpu, &upper, &nItemsPerThread, counter.dataPtr() } );
				oroStream.stop();
				float ms = oroStream.getMs();

				if( 0 < i )
					sumExecMSFind += ms;

				u32 counterValue;
				counter.copyTo( &counterValue );
			}

			//{
			//	LP_ConcurrentCPU lpCpu( NBuckets );
			//	lpGpu.copyTo( &lpCpu );
			//	printf( "Occupancy %f \n", lpCpu.getOccupancy() );
			//}

	#if defined( ENABLE_VARIDATION_GPU )
			LP_ConcurrentCPU lpCpu( NBuckets );
			lpGpu.copyTo( &lpCpu );
			OROASSERT( referenceSet == lpCpu.set(), 0 );
	#endif
		}
		printf( "LP LoadFactor[ %f ] insert find = %f %f \n", loadFactors[i], sumExecMSInsert / NRuns, sumExecMSFind / NRuns );
	}

	printf( "----\n" );

	// Bidirectional Linear Probing execution
	// 1. "insertBLP" kernel - insert random numbers
	// 2. "findBLP" kernel - find inserted numbers and find random numbers
	for( int i = 0; i < NLoadFactors; i++ )
	{
		int nItemsPerThread = 512;

		double insertionRatio = (double)( upper - NBuckets ) / upper;
		int nBlocks = div_round_up( itemsToIssue[i] / nItemsPerThread, BlockSize );

		int NRuns = 4;
		double sumExecMSInsert = 0.0;
		double sumExecMSFind = 0.0;
		for( int i = 0; i < NRuns + 1; i++ )
		{
			BLP_ConcurrentGPU lpGpu( NBuckets );

			{
				OroStopwatch oroStream( sample.getStream() );
				oroStream.start();
				sample.launch1D( "insertBLP", nBlocks, BlockSize, { &lpGpu, &upper, &nItemsPerThread } );
				oroStream.stop();
				float ms = oroStream.getMs();

				if( 0 < i )
					sumExecMSInsert += ms;
			}

			{
				BufferGPU<u32> counter;
				counter.resize( 1 );
				counter.fillZero();
				OroStopwatch oroStream( sample.getStream() );
				oroStream.start();
				sample.launch1D( "findBLP", nBlocks, BlockSize, { &lpGpu, &upper, &nItemsPerThread, counter.dataPtr() } );
				oroStream.stop();
				float ms = oroStream.getMs();

				if( 0 < i )
					sumExecMSFind += ms;

				u32 counterValue;
				counter.copyTo( &counterValue );
			}

#if defined( ENABLE_VARIDATION_GPU )
			LP_ConcurrentCPU lpCpu( NBuckets );
			lpGpu.copyTo( &lpCpu );
			OROASSERT( referenceSet == lpCpu.set(), 0 );
#endif
		}
		printf( "BLP LoadFactor[ %f ] insert find = %f %f \n", loadFactors[i], sumExecMSInsert / NRuns, sumExecMSFind / NRuns );
	}

	// spinlock example
	{
		BufferGPU<u32> counter;
		BufferGPU<u32> mutex;

		counter.resize( 1 );
		mutex.resize( 1 );

		counter.fillZero();
		mutex.fillZero();

		sample.launch1D( "increment", 128, 32,
						 {
							 counter.dataPtr(),
							 mutex.dataPtr(),
						 } );

		OrochiUtils::waitForCompletion();
		u32 n;
		counter.copyTo( &n );

		printf( "n %d\n", n );
	}

	return EXIT_SUCCESS;
}
