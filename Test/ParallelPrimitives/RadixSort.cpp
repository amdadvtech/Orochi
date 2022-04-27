#include <Test/OrochiUtils.h>
#include <Test/ParallelPrimitives/RadixSort.h>
#include <Test/ParallelPrimitives/RadixSortConfigs.h>
#include <numeric>
#include <Test/Stopwatch.h>

//#define PROFILE 1

namespace
{
/// @brief Exclusive scan algorithm on CPU for testing.
/// It copies the count result from the Device to Host before computation, and then copies the offsets back from Host to Device afterward.
/// @param countsGpu The count result in GPU memory. Otuput: The offset.
/// @param offsetsGpu The offsets.
/// @param nWGsToExecute Number of WGs to execute
void exclusiveScanCpu( int* countsGpu, int* offsetsGpu, const int nWGsToExecute ) noexcept
{
	std::vector<int> counts( Oro::BIN_SIZE * nWGsToExecute );
	OrochiUtils::copyDtoH( counts.data(), countsGpu, Oro::BIN_SIZE * nWGsToExecute );
	OrochiUtils::waitForCompletion();

	constexpr auto ENABLE_PRINT{ false };

	if constexpr( ENABLE_PRINT )
	{
		for( int j = 0; j < nWGsToExecute; j++ )
		{
			for( int i = 0; i < Oro::BIN_SIZE; i++ )
			{
				printf( "%d, ", counts[j * Oro::BIN_SIZE + i] );
			}
			printf( "\n" );
		}
	}

	std::vector<int> offsets( Oro::BIN_SIZE * nWGsToExecute );
	std::exclusive_scan( std::cbegin( counts ), std::cend( counts ), std::begin( offsets ), 0 );

	OrochiUtils::copyHtoD( offsetsGpu, offsets.data(), Oro::BIN_SIZE * nWGsToExecute );
	OrochiUtils::waitForCompletion();
}

} // namespace

namespace Oro
{

struct RadixSortImpl
{
	static void printKernelInfo( oroFunction func )
	{
		int a, b, c;
		oroFuncGetAttribute( &a, ORO_FUNC_ATTRIBUTE_NUM_REGS, func );
		oroFuncGetAttribute( &b, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func );
		oroFuncGetAttribute( &c, ORO_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, func );
		printf( "vgpr : shared = %d : %d : %d\n", a, b, c );
	}
	template<typename T>
	static void swap( T& a, T& b )
	{
		T t = a;
		a = b;
		b = t;
	}
};

using I = RadixSortImpl;

RadixSort::RadixSort()
{
	m_flags = (Flag)0;

	if( selectedScanAlgo == ScanAlgo::SCAN_GPU_PARALLEL )
	{
		OrochiUtils::malloc( m_partialSum, m_nWGsToExecute );
		OrochiUtils::malloc( m_isReady, m_nWGsToExecute );
		OrochiUtils::memset( m_isReady, false, m_nWGsToExecute * sizeof( bool ) );
	}
}

RadixSort::~RadixSort()
{
	if( selectedScanAlgo == ScanAlgo::SCAN_GPU_PARALLEL )
	{
		OrochiUtils::free( m_partialSum );
		OrochiUtils::free( m_isReady );
	}
}

void RadixSort::compileKernels( oroDevice device )
{
	constexpr auto kernelPath{ "../Test/ParallelPrimitives/RadixSortKernels.h" };

	printf( "compiling kernels ... \n" );

	std::vector<const char*> opts;
//	opts.push_back( "--save-temps" );
	opts.push_back( "-I ../" );
//	opts.push_back( "-G" );

	oroFunctions[Kernel::COUNT] = OrochiUtils::getFunctionFromFile( device, kernelPath, "CountKernel", &opts );
	if( m_flags & FLAG_LOG ) RadixSortImpl::printKernelInfo( oroFunctions[Kernel::COUNT] );

	oroFunctions[Kernel::COUNT_REF] = OrochiUtils::getFunctionFromFile( device, kernelPath, "CountKernelReference", &opts );
	if( m_flags & FLAG_LOG ) RadixSortImpl::printKernelInfo( oroFunctions[Kernel::COUNT_REF] );

	oroFunctions[Kernel::SCAN_SINGLE_WG] = OrochiUtils::getFunctionFromFile( device, kernelPath, "ParallelExclusiveScanSingleWG", &opts );
	if( m_flags & FLAG_LOG ) RadixSortImpl::printKernelInfo( oroFunctions[Kernel::SCAN_SINGLE_WG] );

	oroFunctions[Kernel::SCAN_PARALLEL] = OrochiUtils::getFunctionFromFile( device, kernelPath, "ParallelExclusiveScanAllWG", &opts );
	if( m_flags & FLAG_LOG ) RadixSortImpl::printKernelInfo( oroFunctions[Kernel::SCAN_PARALLEL] );

	oroFunctions[Kernel::SORT] = OrochiUtils::getFunctionFromFile( device, kernelPath, "SortKernel1", &opts );
	if( m_flags & FLAG_LOG ) RadixSortImpl::printKernelInfo( oroFunctions[Kernel::SORT] );

	oroFunctions[Kernel::SORT_REF] = OrochiUtils::getFunctionFromFile( device, kernelPath, "SortKernelReference", &opts );
	if( m_flags & FLAG_LOG ) RadixSortImpl::printKernelInfo( oroFunctions[Kernel::SORT_REF] );

	oroFunctions[Kernel::SORT_SINGLE_PASS] = OrochiUtils::getFunctionFromFile( device, kernelPath, "SortSinglePassKernel", &opts );
	if( m_flags & FLAG_LOG ) RadixSortImpl::printKernelInfo( oroFunctions[Kernel::SORT_SINGLE_PASS] );
}

void RadixSort::configure( oroDevice device, u32& tempBufferSizeOut )
{
	oroDeviceProp props;
	oroGetDeviceProperties( &props, device );
	const int occupancy = 8; // todo. change me

	const auto newWGsToExecute{ props.multiProcessorCount * occupancy };

	if( newWGsToExecute != m_nWGsToExecute && selectedScanAlgo == ScanAlgo::SCAN_GPU_PARALLEL )
	{
		OrochiUtils::free( m_partialSum );
		OrochiUtils::free( m_isReady );
		OrochiUtils::malloc( m_partialSum, newWGsToExecute );
		OrochiUtils::malloc( m_isReady, newWGsToExecute );
		OrochiUtils::memset( m_isReady, false, newWGsToExecute * sizeof( bool ) );
	}

	m_nWGsToExecute = newWGsToExecute;
	tempBufferSizeOut = BIN_SIZE * m_nWGsToExecute;

	compileKernels( device );
}
void RadixSort::setFlag( Flag flag ) { m_flags = flag; }

void RadixSort::sort( u32* src, u32* dst, int n, int startBit, int endBit, u32* tempBuffer )
{
	u32* s = src;
	u32* d = dst;

	if( n < SINGLE_SORT_WG_SIZE*SINGLE_SORT_N_ITEMS_PER_WI ) //todo. what's the optimal value for SINGLE_SORT_N_ITEMS_PER_WI? there should be a tipping point where single one is inefficient
	{//todo. implement a single pass, single WG sort
		const auto func =  oroFunctions[Kernel::SORT_SINGLE_PASS];
		const void* args[] = { &src, &dst, &n, &startBit, &endBit };
		OrochiUtils::launch1D( func, SINGLE_SORT_WG_SIZE, args, SINGLE_SORT_WG_SIZE );
		OrochiUtils::waitForCompletion();
		return;
	}

	for( int i = startBit; i < endBit; i += N_RADIX )
	{
		sort1pass( s, d, n, i, i + std::min( N_RADIX, endBit - i ), (int*)tempBuffer );

		I::swap( s, d );
	}

	if( s == src )
	{
		OrochiUtils::copyDtoD( dst, src, n );
	}
}

void RadixSort::sort1pass( u32* src, u32* dst, int n, int startBit, int endBit, int* temps )
{
	constexpr bool reference = false;

	// allocate temps
	// clear temps
	// count kernel
	// scan
	// sort

	const int nWIs = WG_SIZE * m_nWGsToExecute;
	int nItemsPerWI = ( n + ( nWIs - 1 ) ) / nWIs;
	if( m_flags & FLAG_LOG )
	{
		printf( "nWGs: %d\n", m_nWGsToExecute );
		printf( "nNItemsPerWI: %d\n", nItemsPerWI );
	}

	float t[3] = {0.f};
	{
		Stopwatch sw; sw.start();
		const auto func{ reference ? oroFunctions[Kernel::COUNT_REF] : oroFunctions[Kernel::COUNT] };
		const void* args[] = { &src, &temps, &n, &nItemsPerWI, &startBit, &m_nWGsToExecute };
		OrochiUtils::launch1D( func, WG_SIZE * m_nWGsToExecute, args, WG_SIZE );
#if defined(PROFILE)
		OrochiUtils::waitForCompletion();
		sw.stop();
		t[0] = sw.getMs();
#endif
	}

	{
		Stopwatch sw; sw.start();
		switch( selectedScanAlgo )
		{
		case ScanAlgo::SCAN_CPU:
		{
			exclusiveScanCpu( temps, temps, m_nWGsToExecute );
		}
		break;

		case ScanAlgo::SCAN_GPU_SINGLE_WG:
		{
			const void* args[] = { &temps, &temps, &m_nWGsToExecute };
			OrochiUtils::launch1D( oroFunctions[Kernel::SCAN_SINGLE_WG], WG_SIZE * m_nWGsToExecute, args, WG_SIZE );
		}
		break;

		case ScanAlgo::SCAN_GPU_PARALLEL:
		{
			const void* args[] = { &temps, &temps, &m_partialSum, &m_isReady };
			OrochiUtils::launch1D( oroFunctions[Kernel::SCAN_PARALLEL], SCAN_WG_SIZE * m_nWGsToExecute, args, SCAN_WG_SIZE );
		}
		break;

		default:
			exclusiveScanCpu( temps, temps, m_nWGsToExecute );
			break;
		}
#if defined(PROFILE)
		OrochiUtils::waitForCompletion();
		sw.stop();
		t[1] = sw.getMs();
#endif
	}

	{
		Stopwatch sw; sw.start();
		const auto func{ reference ? oroFunctions[Kernel::SORT_REF] : oroFunctions[Kernel::SORT] };
		const void* args[] = { &src, &dst, &temps, &n, &nItemsPerWI, &startBit, &m_nWGsToExecute };
		OrochiUtils::launch1D( func, WG_SIZE * m_nWGsToExecute, args, WG_SIZE );
#if defined(PROFILE)
		OrochiUtils::waitForCompletion();
		sw.stop();
		t[2] = sw.getMs();
#endif
	}
#if defined(PROFILE)
	printf("%3.2f, %3.2f, %3.2f\n", t[0], t[1], t[2]);
#endif
}

}; // namespace Oro
