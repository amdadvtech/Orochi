#include <common/Common.h>
#include <iostream>
#include <memory>
#include <set>
#include <random>
#include <mutex>
#include <Windows.h>
#include <ppl.h>

#include <08_LinearProbing/LP.h>

// Reference
// [0] Ordered hash tables ( original idea of BLP, very old )
// [1] A Concurrent Bidirectional Linear Probing Algorithm
// [2] A Parallel Compact Hash Table ( after [1] )

// very nice code example:
// https://github.com/senderista/hashtable-benchmarks



class LP
{
  public:
	enum
	{
		OCCUPIED_BIT = 1 << 31,
		VALUE_MASK = ~OCCUPIED_BIT,
	};

	LP( int n ) : m_table( n ) {}

	int home( uint32_t k ) const { return hash( k ) % m_table.size(); }

	enum InsertionResult
	{
		INSERTED,
		FOUND,
		OUT_OF_MEMORY
	};
	// k must be less than equal 0x7FFFFFFF
	InsertionResult insert( uint32_t k )
	{
		uint32_t h = home( k );
		for( int i = 0; i < m_table.size(); i++ )
		{
			int location = ( h + i ) % m_table.size();

			if( ( m_table[location] & OCCUPIED_BIT ) == 0 )
			{
				m_table[location] = k | OCCUPIED_BIT;
				return INSERTED;
			}
			else if( ( m_table[location] & VALUE_MASK ) == k )
			{
				return FOUND;
			}
		}
		return OUT_OF_MEMORY;
	}
	bool find( uint32_t k ) const
	{
		uint32_t h = home( k );
		for( int i = 0; i < m_table.size(); i++ )
		{
			int location = ( h + i ) % m_table.size();
			if( ( m_table[location] & OCCUPIED_BIT ) == 0 )
			{
				return false;
			}
			else if( m_table[location] == ( k | OCCUPIED_BIT ) )
			{
				return true;
			}
		}
		return false;
	}
	std::set<uint32_t> set() const
	{
		std::set<uint32_t> s;
		for( auto value : m_table )
		{
			if( value & OCCUPIED_BIT )
			{
				s.insert( value & VALUE_MASK );
			}
		}
		return s;
	}

	void print()
	{
		printf( "data=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			if( m_table[i] & OCCUPIED_BIT )
			{
				printf( "%03d, ", m_table[i] & VALUE_MASK );
			}
			else
			{
				printf( "---, " );
			}
		}
		printf( "\n" );

		printf( "home=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			if( m_table[i] & OCCUPIED_BIT )
			{
				printf( "%03d, ", home( m_table[i] & VALUE_MASK ) );
			}
			else
			{
				printf( "---, " );
			}
		}
		printf( "\n" );
	}
	std::vector<uint32_t> m_table;
};

class StdSet_Concurrent
{
  public:
	StdSet_Concurrent( int n = 0 ){
	}
	void insert( u32 k )
	{
		std::lock_guard lock( m_mu );
		m_set.insert( k );
	}

	bool find( u32 k ) const
	{
		std::lock_guard lock( m_mu );
		return m_set.count( k ) != 0;
	}
	std::set<u32> set() const
	{
		std::lock_guard lock( m_mu );
		return m_set;
	}
	std::set<u32> m_set;
	mutable std::mutex m_mu;

	// int m_table[1];
};

class RH
{
  public:
	enum
	{
		OCCUPIED_BIT = 1 << 31,
		VALUE_MASK = ~OCCUPIED_BIT,
	};

	RH( int n ) : m_table( n ) {}

	uint32_t home( uint32_t h ) const { return (uint64_t)h * m_table.size() / ( (uint64_t)UINT_MAX + 1 ); }

	// probe sequence lengths
	uint32_t psl( int i, uint32_t h )
	{
		/*
		PSL =
		 2  3  4     1
		[ ][ ][ ][h][ ]
		*/
		int homeLocation = (int)home( h );

		if( i < homeLocation )
		{
			// e.g. h 3 - 5 = -2
			homeLocation = homeLocation - (int)m_table.size();
		}

		return i - homeLocation;
	}

	// k must be less than equal 0x7FFFFFFF
	int insert( uint32_t k )
	{
		uint32_t h = home( hash( k ) );
		int PSL = 0;
		for( int i = 0; i < m_table.size(); i++ )
		{
			int location = ( h + i ) % m_table.size();

			if( ( m_table[location] & OCCUPIED_BIT ) == 0 )
			{
				m_table[location] = k | OCCUPIED_BIT;
				return location;
			}
			else if( ( m_table[location] & VALUE_MASK ) == k )
			{
				return location;
			}
			int thePSL = psl( location, hash( m_table[location] & VALUE_MASK ) );
			if( thePSL < PSL )
			{
				uint32_t theValue = m_table[location] & VALUE_MASK;
				m_table[location] = k | OCCUPIED_BIT;

				PSL = thePSL;
				k = theValue;
			}
			PSL++;
		}
		return -1;
	}
	bool find( uint32_t k ) const
	{
		uint32_t h = home( hash( k ) );
		for( int i = 0; i < m_table.size(); i++ )
		{
			int location = ( h + i ) % m_table.size();
			if( ( m_table[location] & OCCUPIED_BIT ) == 0 )
			{
				return false;
			}
			else if( m_table[location] == ( k | OCCUPIED_BIT ) )
			{
				return true;
			}
		}
		return false;
	}
	std::set<uint32_t> set() const
	{
		std::set<uint32_t> s;
		for( auto value : m_table )
		{
			if( value & OCCUPIED_BIT )
			{
				s.insert( value & VALUE_MASK );
			}
		}
		return s;
	}

	void print()
	{
		printf( "data=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			if( m_table[i] & OCCUPIED_BIT )
			{
				printf( "%03d, ", m_table[i] & VALUE_MASK );
			}
			else
			{
				printf( "---, " );
			}
		}
		printf( "\n" );

		printf( "home=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			if( m_table[i] & OCCUPIED_BIT )
			{
				printf( "%03d, ", home( m_table[i] & VALUE_MASK ) );
			}
			else
			{
				printf( "---, " );
			}
		}
		printf( "\n" );

		printf( "PSL =" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			if( m_table[i] & OCCUPIED_BIT )
			{
				printf( "%03d, ", psl( i, hash( m_table[i] & VALUE_MASK ) ) );
			}
			else
			{
				printf( "---, " );
			}
		}
		printf( "\n" );
	}
	std::vector<uint32_t> m_table;
};
class BLP
{
  public:
	enum
	{
		OCCUPIED_BIT = 1 << 31,
		VALUE_MASK = ~OCCUPIED_BIT,
	};

	BLP( int n ) : m_table( n ) {}

	uint32_t home( uint32_t k ) const
	{
		uint32_t v = hash( k );
		return (uint64_t)v * m_table.size() / ( (uint64_t)UINT_MAX + 1 );
	}

	// k must be less than equal 0x7FFFFFFF
	int insert( uint32_t k )
	{
		uint32_t h = home( k );
		if( ( m_table[h] & OCCUPIED_BIT ) == 0 )
		{
			m_table[h] = k | OCCUPIED_BIT;
			return h;
		}

		if( m_table[h] == ( k | OCCUPIED_BIT ) )
		{
			return h;
		}

		uint32_t hashK = hash( k );

		bool moveTowardLeft = hash( m_table[h] & VALUE_MASK ) < hashK;

		for( int iter = 0; iter < 2; iter++ )
		{
			int j = h;
			if( moveTowardLeft )
			{
				// find
				if( iter == 0 )
				{
					while( j + 1 < m_table.size() && m_table[j + 1] & OCCUPIED_BIT && hash( m_table[j + 1] & VALUE_MASK ) <= hashK )
					{
						j++;
					}
					if( ( m_table[j] & VALUE_MASK ) == k )
					{
						return j;
					}
					j = h;
				}

				// example: hash(k) = 8
				// 7 < hash(k)
				// [ ][ ][3][6][7][ ]
				//     |<-------j
				// The elements are too right-shifed.
				// find empty location on the left
				while( 0 < j && m_table[j] & OCCUPIED_BIT )
				{
					j--;
				}

				if( m_table[j] & OCCUPIED_BIT )
				{
					// No empty space in this dir. Try other direction
					moveTowardLeft = !moveTowardLeft;
					continue;
				}

				// move toword left while T[j+1] < hashK
				//     +--+
				//     v  |
				// [ ][3][3][6][7][ ]
				//     j
				while( j + 1 < m_table.size() && ( m_table[j + 1] & OCCUPIED_BIT ) && hash( m_table[j + 1] & VALUE_MASK ) < hashK )
				{
					m_table[j] = m_table[j + 1];
					j++;
				}

				// [ ][3][6][7][7][ ]
				//              j
			}
			else // hashK <= hash( m_table[h] & VALUE_MASK );
			{
				// find
				if( iter == 0 )
				{
					while( 0 <= j - 1 && m_table[j - 1] & OCCUPIED_BIT && hashK <= hash( m_table[j - 1] & VALUE_MASK ) )
					{
						j--;
					}
					if( ( m_table[j] & VALUE_MASK ) == k )
					{
						return j;
					}
					j = h;
				}

				// example: hash(k) = 5
				// hash(k) < 6
				// [ ][ ][ ][2][6][7][8][ ]
				//              h
				// The elements are too left-shifed.
				// find empty location on the right
				while( j + 1 < m_table.size() && m_table[j] & OCCUPIED_BIT )
				{
					j++;
				}

				if( m_table[j] & OCCUPIED_BIT )
				{
					// No empty space in this dir. Try other direction
					moveTowardLeft = !moveTowardLeft;
					continue;
				}

				// move toword right while hashK < T[j-1]
				//                    +--+
				//                    |  v
				// [ ][ ][ ][2][6][7][8][8]
				//                       j
				while( 0 <= j - 1 && ( m_table[j - 1] & OCCUPIED_BIT ) && hashK < hash( m_table[j - 1] & VALUE_MASK ) )
				{
					m_table[j] = m_table[j - 1];
					j--;
				}

				// [ ][ ][ ][2][6][6][7][8]
				//              j
			}

			m_table[j] = k | OCCUPIED_BIT;
			return j;
		}
		return -1;
	}
	bool find( uint32_t k ) const
	{
		uint32_t h = home( k );
		if( ( m_table[h] & OCCUPIED_BIT ) == 0 )
		{
			return false;
		}
		if( m_table[h] == ( k | OCCUPIED_BIT ) )
		{
			return true;
		}

		uint32_t hashK = hash( k );
		bool moveTowardLeft = hash( m_table[h] & VALUE_MASK ) < hashK;
		int j = h;
		if( moveTowardLeft )
		{
			while( j + 1 < m_table.size() && m_table[j + 1] & OCCUPIED_BIT && hash( m_table[j + 1] & VALUE_MASK ) <= hashK )
			{
				j++;
			}
			if( m_table[j] == ( k | OCCUPIED_BIT ) )
			{
				return true;
			}
		}
		else
		{
			while( 0 <= j - 1 && m_table[j - 1] & OCCUPIED_BIT && hashK <= hash( m_table[j - 1] & VALUE_MASK ) )
			{
				j--;
			}
			if( m_table[j] == ( k | OCCUPIED_BIT ) )
			{
				return true;
			}
		}
		return false;
	}

	std::set<uint32_t> set() const
	{
		std::set<uint32_t> s;
		for( auto value : m_table )
		{
			if( value & OCCUPIED_BIT )
			{
				s.insert( value & VALUE_MASK );
			}
		}
		return s;
	}

	void print()
	{
		printf( "data=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			if( m_table[i] & OCCUPIED_BIT )
			{
				printf( "%02d, ", m_table[i] & ~OCCUPIED_BIT );
			}
			else
			{
				printf( "--, " );
			}
		}
		printf( "\nhome=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			printf( "%02d, ", home( m_table[i] & ~OCCUPIED_BIT ) );
		}
		printf( "\n" );

		printf( "hash=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			printf( "%x, ", hash( m_table[i] & VALUE_MASK ) );
		}
		printf( "\n" );
	}
	std::vector<uint32_t> m_table;
};

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
			s.insert( v );
			lp.insert( v );
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
	printf( "test / %s OK\n", typeid( T ).name() );
}

template<class T>
void runPerfTest( )
{
	int NBuckets = 100000;
	int Numbers = 1000000;
	double loadFactor = 0.75;

	float timeInsert = 0;
	float timeFind = 0;

	int nfound = 0;
	splitmix64 rnd;
	for( int i = 0; i < 1000; i++ )
	{
		T lp( NBuckets );

		Stopwatch sw;
		sw.start();

		for( int j = 0; j < NBuckets * loadFactor; j++ )
		{
			uint32_t v = rnd.next() % Numbers;
			lp.insert( v );
		}

		sw.split();

		for( int i = 0; i < NBuckets; i++ )
		{
			uint32_t v = rnd.next() % Numbers;
			bool found = lp.find( v );
			if( found )
			{
				nfound++;
			}
		}

		sw.stop();

		float times[2];
		sw.getMs( times, 2 );
		timeInsert += times[0];
		timeFind += times[1];
	}

	printf( "%s insert %f ms, find %f ms %d\n", typeid( T ).name(), timeInsert, timeFind, nfound );
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
	printf( "test con / %s OK\n", typeid( T ).name() );
}
template <class T>
void runConcurrentPerfTest()
{ 
	int NThreads = 32;
	int NBuckets = 100000;
	int Numbers = 10000000;
	double loadFactor = 0.75;


	int nfound = 0;

	float timeInsertion = 0;
	float timeFind = 0;

	Stopwatch sw;
	sw.start();
	
	int a = 0;

	for (int k = 0; k < 512; k++)
	{
		T storage( NBuckets );
		for( int i = 0; i < NThreads; i++ )
		{
			int nItemPerThread = NBuckets * loadFactor / NThreads;
			concurrency::parallel_for( 0, NThreads, [k, nItemPerThread, Numbers, &storage](int index) 
			{
				splitmix64 rnd;
				rnd.x = k * 100000 + index;
				for( int j = 0; j < nItemPerThread; j++ )
				{
					uint32_t v = rnd.next() % Numbers;
					storage.insert( v );
				}
			} );
		}
		a += storage.m_table[0];
	}
	sw.stop();

	printf( "con / %s insertion %f ms %d\n", typeid( T ).name(), sw.getMs(), a );
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
			//"--gpu-architecture=compute_75"
		};
		oroFunction f = m_utils.getFunctionFromFile( m_device, "../08_LinearProbing/Kernels.h", function, &opts );
		oroModuleLaunchKernel( f, gridDimX, 1, 1, blockDimX, 1, 1, 0, m_stream, std::vector<void*>( args ).data(), 0 );
	}
	oroStream getStream() { return m_stream; }
	oroStream m_stream = 0;
};



int main( int argc, char** argv )
{
	// 6
	//{
	//	BLP_Concurrent blp( 10 );
	//	blp.insert( 56 );
	//	blp.print();
	//	blp.insert( 48 );
	//	blp.print();
	//	blp.insert( 69 );
	//	blp.print();

	//	printf( "" );
	//}

	//{
	//	int NBuckets = 1000;
	//	int Numbers = 10000;
	//	splitmix64 rnd;

	//	for( int i = 0; i < 10000; i++ )
	//	{
	//		BLP blp( NBuckets );
	//		BLP_Concurrent blpc( NBuckets );
	//		for( int j = 0; j < NBuckets * 1.75; j++ )
	//		{
	//			uint32_t v = rnd.next() % Numbers;
	//			blp.insert( v );
	//			blpc.insert( v );
	//		}

	//		for (int i = 0; i < NBuckets; i++)
	//		{
	//			OROASSERT( blp.m_table[i] == blpc.m_table[i], 0 );
	//		}
	//	}
	//}

	//runTest<BLP_Concurrent>();
	//runConcurrentPerfTest<LP_ConcurrentCPU>();
	//runConcurrentPerfTest<BLP_Concurrent>();

	//runPerfTest<LP_ConcurrentCPU>();
	//runPerfTest<BLP_Concurrent>();
	//blp.insert( 69 ); // 6 on the left side
	//blp.print();
	//blp.insert( 158 );
	//blp.insert( 90 );
	//blp.insert( 35 );
	//blp.insert( 179 );
	// blp.print();
	// 

	printf( "--- Test ---\n" );
	runTest<LP>();
	runTest<BLP>();
	runTest<RH>();
	runTest<LP_ConcurrentCPU>();
	runTest<BLP_ConcurrentCPU>();

	runConcurrentTest<LP_ConcurrentCPU>();
	runConcurrentTest<BLP_ConcurrentCPU>();

	printf( "--- perf ---\n" );
	{
		runConcurrentPerfTest<LP_ConcurrentCPU>();
		runConcurrentPerfTest<BLP_ConcurrentCPU>();

		runPerfTest<LP>();
		runPerfTest<LP_ConcurrentCPU>();
		runPerfTest<BLP_ConcurrentCPU>();
		runPerfTest<RH>();
		runPerfTest<BLP>();
	}

	LPSample sample;

	int BlockSize = 32;
	int NBuckets = 100000000;
	int upper    = 200000000;

// takes some time
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

	const double loadFactors[] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
	const int NLoadFactors = sizeof( loadFactors ) / sizeof( loadFactors[0] );
#if 1
	uint64_t itemsToIssue[NLoadFactors] =
	{ 
		10258472,
		21073696,
		32505788,
		44630383,
		57537883,
		71337585,
		86160429,
		102168786,
		119568562,
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
		printf( "LP LoadFactor[ %f ] insert - find = %f - %f \n", loadFactors[i], sumExecMSInsert / NRuns, sumExecMSFind / NRuns );
	}

	printf( "----\n" );

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
		printf( "BLP LoadFactor[ %f ] insert - find = %f - %f \n", loadFactors[i], sumExecMSInsert / NRuns, sumExecMSFind / NRuns );
	}

//
//	for( int i = 0; i < 4; i++ )
//	{
//		BLP_ConcurrentGPU blpGpu( NBuckets );
//
//		{
//			OroStopwatch oroStream( sample.getStream() );
//			oroStream.start();
//			sample.launch1D( "insertBLP", nBlocks, BlockSize, { &blpGpu, &upper, &nItemsPerThread } );
//			oroStream.stop();
//			float ms = oroStream.getMs();
//			printf( "insertBLP %f ms \n", ms );
//		}
//
//		//BLP_Concurrent<true> lpCpu;
//		//blpGpu.copyTo( &lpCpu );
//		//printf( "occupancy %f \n", lpCpu.getOccupancy() );
//
//		{
//			BufferGPU<u32> counter;
//			counter.resize( 1 );
//			counter.fillZero();
//			OroStopwatch oroStream( sample.getStream() );
//			oroStream.start();
//			sample.launch1D( "findBLP", nBlocks, BlockSize, { &blpGpu, &upper, &nItemsPerThread, counter.dataPtr() } );
//			oroStream.stop();
//			float ms = oroStream.getMs();
//
//			u32 counterValue;
//			counter.copyTo( &counterValue );
//			printf( "findBLP %f ms, counter = %d \n", ms, counterValue );
//		}
//#if defined( ENABLE_VARIDATION_GPU )
//		BLP_ConcurrentCPU blpCpu( NBuckets );
//		blpGpu.copyTo( &blpCpu );
//		blpCpu.set();
//		OROASSERT( referenceSet == blpCpu.set(), 0 );
//#endif
//	}

	// lock 
	//{
	//	BufferGPU<u32> counter;
	//	BufferGPU<u32> mutex;

	//	counter.resize( 1 );
	//	mutex.resize( 1 );

	//	counter.fillZero();
	//	mutex.fillZero();

	//	sample.launch1D( "increment", 128, 32,
	//					 {
	//						 counter.dataPtr(),
	//						 mutex.dataPtr(),
	//					 } );

	//	OrochiUtils::waitForCompletion();
	//	u32 n;
	//	counter.copyTo( &n );

	//	printf( "n %d\n", n );
	//}

	//LP_Concurrent<true> lpCpu;
	//lpGpu.copyTo( &lpCpu );

	//for (auto v : lpCpu.set())
	//{
	//	printf( "%d\n", v );
	//}

	//sample.getOroUtil()->getFunctionFromFile( sample.getOroUtil(), "../01_Reduction/Kernels.h", kernelName, &opts );
	//const void* args[] = { &size, d_input.address(), d_output.address() };
	//for( u32 i = 0; i < RunCount; ++i )
	//{
	//	int h_output = 0;
	//	for( u32 j = 0; j < size; ++j )
	//	{
	//		h_input[j] = distribution( generator );
	//		h_output += h_input[j];
	//	}
	//	d_input.copyFromHost( h_input.data(), size );

	//	OrochiUtils::memset( d_output.ptr(), 0, sizeof( int ) );
	//	OrochiUtils::waitForCompletion();
	//	sw.start();

	//	OrochiUtils::launch1D( func, size, args, BlockSize, BlockSize * sizeof( int ) );
	//	OrochiUtils::waitForCompletion();
	//	sw.stop();

	//	OROASSERT( h_output == d_output.getSingle(), 0 );

	//for(int i = 0 ; i < 10 ; i++)
	//{
	//	int NBuckets = 1000;
	//	int Numbers = 10000;
	//	double loadFactor = 0.75;

	//	Stopwatch sw;
	//	sw.start();

	//	int nfound = 0;
	//	splitmix64 rnd;
	//	for( int i = 0; i < 100000; i++ )
	//	{
	//		BLPZeroEmptyBranchless lp( NBuckets );
	//		for( int j = 0; j < NBuckets * loadFactor; j++ )
	//		{
	//			uint32_t v = rnd.next() % Numbers;
	//			lp.insert( v );
	//		}

	//		for( int i = 0; i < NBuckets * loadFactor; i++ )
	//		{
	//			uint32_t v = rnd.next() % Numbers;
	//			bool found = lp.find( v ) != -1;
	//			if( found )
	//			{
	//				nfound++;
	//			}
	//		}
	//	}

	//	sw.stop();
	//	printf( "%s %f ms, %d\n", typeid( BLPZeroEmptyBranchless ).name(), sw.getMs(), nfound );
	//}
	return EXIT_SUCCESS;
}
