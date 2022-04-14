#include <Test/OrochiUtils.h>
#include <Test/ParallelPrimitives/RadixSort.h>
#include <Test/ParallelPrimitives/RadixSortConfigs.h>
#include <numeric>

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

	compileKernels();

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

void RadixSort::compileKernels()
{
	constexpr auto kernelPath{ "../Test/ParallelPrimitives/RadixSortKernels.h" };

	printf( "compiling kernels ... \n" );

	oroFunctions[Kernel::COUNT] = OrochiUtils::getFunctionFromFile( kernelPath, "CountKernel", 0 );
	if( m_flags & FLAG_LOG ) RadixSortImpl::printKernelInfo( oroFunctions[Kernel::COUNT] );

	oroFunctions[Kernel::COUNT_REF] = OrochiUtils::getFunctionFromFile( kernelPath, "CountKernelReference", 0 );
	if( m_flags & FLAG_LOG ) RadixSortImpl::printKernelInfo( oroFunctions[Kernel::COUNT_REF] );

	oroFunctions[Kernel::SCAN_SINGLE_WG] = OrochiUtils::getFunctionFromFile( kernelPath, "ParallelExclusiveScanSingleWG", 0 );
	if( m_flags & FLAG_LOG ) RadixSortImpl::printKernelInfo( oroFunctions[Kernel::SCAN_SINGLE_WG] );

	oroFunctions[Kernel::SCAN_PARALLEL] = OrochiUtils::getFunctionFromFile( kernelPath, "ParallelExclusiveScanAllWG", 0 );
	if( m_flags & FLAG_LOG ) RadixSortImpl::printKernelInfo( oroFunctions[Kernel::SCAN_PARALLEL] );

	oroFunctions[Kernel::SORT] = OrochiUtils::getFunctionFromFile( kernelPath, "SortKernel2", 0 );
	if( m_flags & FLAG_LOG ) RadixSortImpl::printKernelInfo( oroFunctions[Kernel::SORT] );

	oroFunctions[Kernel::SORT_REF] = OrochiUtils::getFunctionFromFile( kernelPath, "SortKernelReference", 0 );
	if( m_flags & FLAG_LOG ) RadixSortImpl::printKernelInfo( oroFunctions[Kernel::SORT_REF] );
}

void RadixSort::configure( oroDevice device, u32& tempBufferSizeOut )
{
	oroDeviceProp props;
	oroGetDeviceProperties( &props, device );
	const int occupancy = 2; // todo. change me

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
}
void RadixSort::setFlag( Flag flag ) { m_flags = flag; }

void RadixSort::sort( u32* src, u32* dst, int n, int startBit, int endBit, u32* tempBuffer )
{
	u32* s = src;
	u32* d = dst;
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

	{
		const auto func{ reference ? oroFunctions[Kernel::COUNT_REF] : oroFunctions[Kernel::COUNT] };
		const void* args[] = { &src, &temps, &n, &nItemsPerWI, &startBit, &m_nWGsToExecute };
		OrochiUtils::launch1D( func, WG_SIZE * m_nWGsToExecute, args, WG_SIZE );
		//		OrochiUtils::waitForCompletion();
	}

	{
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
			OrochiUtils::launch1D( oroFunctions[Kernel::SCAN_PARALLEL], WG_SIZE * m_nWGsToExecute, args, WG_SIZE );
		}
		break;

		default:
			exclusiveScanCpu( temps, temps, m_nWGsToExecute );
			break;
		}
	}

	{
		const auto func{ reference ? oroFunctions[Kernel::SORT_REF] : oroFunctions[Kernel::SORT] };
		const void* args[] = { &src, &dst, &temps, &n, &nItemsPerWI, &startBit, &m_nWGsToExecute };
		OrochiUtils::launch1D( func, WG_SIZE * m_nWGsToExecute, args, WG_SIZE );
		//		OrochiUtils::waitForCompletion();
	}
}

}; // namespace Oro
