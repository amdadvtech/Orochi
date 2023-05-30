#include <09_RadixSort/Configs.h>
#include <common/Common.h>
#include <string>
#include <vector>
#include <iostream>

namespace
{
template<typename T>
void* arg_cast( T* const* ptr ) noexcept
{
	return reinterpret_cast<void*>( const_cast<T**>( ptr ) );
}

template<typename T>
inline T getRandom( const T minV, const T maxV )
{
	double r = std::min( static_cast<double>( RAND_MAX ) - 1, static_cast<double>( rand() ) ) / RAND_MAX;
	T range = maxV - minV;
	return static_cast<T>( minV + r * range );
}

} // namespace

class RadixSortSample : public Sample
{
  public:
	RadixSortSample()
	{
		partial_sum.resize( num_workgroup_to_execute );
		is_ready.resize( num_workgroup_to_execute );
		OrochiUtils::memset( is_ready.ptr(), false, num_workgroup_to_execute * sizeof( bool ) );

		tmp_buffer.resize( BIN_SIZE * num_workgroup_to_execute );

		compile();
	}

	void run( int testSize, const int testBits = 32, const int nRuns = 1 )
	{
		printf( "Start Testing !!!\n" );

		srand( 123 );

		std::vector<u32> srcKey( testSize );
		for( int i = 0; i < testSize; ++i )
		{
			srcKey[i] = getRandom( 0u, static_cast<u32>( ( 1ull << static_cast<u64>( testBits ) ) - 1 ) );
		}

		Oro::GpuMemory<u32> gpu_src;
		gpu_src.resize( testSize );

		Oro::GpuMemory<u32> gpu_dst;
		gpu_dst.resize( testSize );

		for( int i = 0; i < nRuns; i++ )
		{
			gpu_src.copyFromHost( srcKey.data(), testSize );
			OrochiUtils::waitForCompletion();

			sort( gpu_src, gpu_dst, testSize );
			OrochiUtils::waitForCompletion();
		}

		std::vector<u32> dstKey( testSize );
		OrochiUtils::copyDtoH( dstKey.data(), gpu_dst.ptr(), testSize );

		std::vector<u32> indexHelper( testSize );
		std::iota( std::begin( indexHelper ), std::end( indexHelper ), 0U );

		std::stable_sort( std::begin( indexHelper ), std::end( indexHelper ), [&]( const auto indexA, const auto indexB ) noexcept { return srcKey[indexA] < srcKey[indexB]; } );

		const auto rearrange = []( auto& targetBuffer, const auto& indexBuffer ) noexcept
		{
			std::vector<u32> tmpBuffer( std::size( targetBuffer ) );

			for( auto i = 0UL; i < std::size( targetBuffer ); ++i )
			{
				tmpBuffer[i] = targetBuffer[indexBuffer[i]];
			}

			targetBuffer = std::move( tmpBuffer );
		};

		rearrange( srcKey, indexHelper );

		for( int i = 0; i < testSize; i++ )
		{
			if( dstKey[i] != srcKey[i] )
			{
				printf( "fail at %d\n", i );
				__debugbreak();
				break;
			}
		}

		printf( "passed: %3.2fK keys\n", testSize / 1000.f );
	}

  private:
	void compile() noexcept
	{
		struct Record
		{
			std::string kernelName;
			Kernel kernelType;
		};

		const std::vector<Record> records{
			{ "CountKernel", Kernel::COUNT },
			{ "ParallelExclusiveScan", Kernel::SCAN },
			{ "SortKernel", Kernel::SORT },
		};

		std::vector<const char*> opts;
		opts.push_back( "-I../" );

		constexpr auto kernel_path{ "../09_RadixSort/Kernels.h" };

		for( const auto& record : records )
		{
			oroFunctions[record.kernelType] = m_utils.getFunctionFromFile( m_device, kernel_path, record.kernelName.c_str(), &opts );
		}
	}

	void sort1pass( Oro::GpuMemory<u32>& src, Oro::GpuMemory<u32>& dst, int n, int startBit ) noexcept
	{
		int nWGsToExecute = num_workgroup_to_execute;

		const int nWIs = WG_SIZE * nWGsToExecute;
		int nItemsPerWI = ( n + ( nWIs - 1 ) ) / nWIs;

		// Adjust nItemsPerWI to be dividable by SORT_N_ITEMS_PER_WI.
		nItemsPerWI = ( std::ceil( static_cast<double>( nItemsPerWI ) / SORT_N_ITEMS_PER_WI ) ) * SORT_N_ITEMS_PER_WI;

		int nItemPerWG = nItemsPerWI * WG_SIZE;

		auto t1 = src.getData();


		{
			const void* args[] = { arg_cast( src.address() ), arg_cast( tmp_buffer.address() ), &n, &nItemPerWG, &startBit, &nWGsToExecute };
			OrochiUtils::launch1D( oroFunctions[Kernel::COUNT], COUNT_WG_SIZE * nWGsToExecute, args, COUNT_WG_SIZE, 0, default_stream );
		}

		auto t2 = tmp_buffer.getData();

		{
			const void* args[] = { arg_cast( tmp_buffer.address() ), arg_cast( tmp_buffer.address() ), arg_cast( partial_sum.address() ), arg_cast( is_ready.address() ) };
			OrochiUtils::launch1D( oroFunctions[Kernel::SCAN], SCAN_WG_SIZE * nWGsToExecute, args, SCAN_WG_SIZE, 0, default_stream );
		}

		auto t3 = tmp_buffer.getData();

		{
			const void* args[] = { arg_cast( src.address() ), arg_cast( dst.address() ), arg_cast( tmp_buffer.address() ), &n, &nItemsPerWI, &startBit, &nWGsToExecute };
			OrochiUtils::launch1D( oroFunctions[Kernel::SORT], SORT_WG_SIZE * nWGsToExecute, args, SORT_WG_SIZE, 0, default_stream );
		}
	}

	void sort( Oro::GpuMemory<u32>& src, Oro::GpuMemory<u32>& dst, int n ) noexcept
	{
		auto* s{ &src };
		auto* d{ &dst };

		for( int i = start_bit; i < end_bit; i += N_RADIX )
		{
			sort1pass( *s, *d, n, i );
			std::swap( s, d );
		}

		auto t1 = src.getData();
		auto t2 = dst.getData();

		if( s == &src )
		{
			OrochiUtils::copyDtoDAsync( dst.ptr(), src.ptr(), n, default_stream );
		}
	}

	static constexpr oroStream default_stream{ 0 };

	static constexpr auto start_bit{ 0 };
	static constexpr auto end_bit{ 32 };

	static constexpr int num_workgroup_to_execute{ 4 };

	Oro::GpuMemory<int> partial_sum;
	Oro::GpuMemory<bool> is_ready;
	Oro::GpuMemory<int> tmp_buffer;

	enum class Kernel
	{
		COUNT,
		SCAN,
		SORT,
	};

	std::unordered_map<Kernel, oroFunction> oroFunctions;
};

int main()
{
	RadixSortSample sample;

	const int testSize = 16 * 10;
	sample.run( testSize );
}