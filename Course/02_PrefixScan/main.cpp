#include <common/Common.h>

class PrefixScanSample : public Sample
{
  public:
	void run( u32 size ) 
	{ 
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution( 0, 16 );

		Oro::GpuMemory<int> d_input( size );
		Oro::GpuMemory<int> d_output( size );
		Oro::GpuMemory<int> d_counter( 1 );
		Oro::GpuMemory<int> d_sum( 1 );

		std::vector<int> h_input( size );
		std::vector<int> h_output( size );
		h_output[0] = 0;
		for( u32 i = 0; i < size; ++i )
		{
			h_input[i] = distribution( generator );
			if( i == 0 )
				h_output[i] = h_input[i];
			else
				h_output[i] = h_input[i] + h_output[i - 1];
		}
		d_input.copyFromHost( h_input.data(), size );

		std::vector<const char*> opts;
		opts.push_back( "-I../" );

		Stopwatch sw;
		auto test = [&]( const char* kernelName )
		{
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../02_PrefixScan/Kernels.h", kernelName, &opts );
			const void* args[] = { &size, d_input.address(), d_output.address(), d_sum.address(), d_counter.address() };
			for( u32 i = 0; i < RunCount; ++i )
			{
				OrochiUtils::memset( d_counter.ptr(), 0, sizeof( int ) );
				OrochiUtils::memset( d_sum.ptr(), 0, sizeof( int ) );
				OrochiUtils::waitForCompletion();
				sw.start();

				OrochiUtils::launch1D( func, size, args, BlockSize, BlockSize * sizeof( float ) );
				OrochiUtils::waitForCompletion();
				sw.stop();

				std::vector<int> output = d_output.getData();
				for( u32 j = 0; j < size; ++j )
					OROASSERT( h_output[j] == output[j], 0 );

				float ms = sw.getMs();
				float gItemsS = static_cast<float>( size ) / 1000.0f / 1000.0f / ms;
				float mItems = size / 1000.0f / 1000.0f;
				std::cout << std::setprecision( 2 ) << mItems << "M items scanned in " << ms << " ms (" << gItemsS << " GItems/s), result " 
					<< "[" << kernelName << "] " << std::endl;
			}
		};

		test( "ScanDeviceBlellochKernel" );
		test( "ScanDeviceHillisSteeleKernel" );
	}
};

int main( int argc, char** argv )
{
	PrefixScanSample sample;
	sample.run( 16 * 1000 * 1 );
	sample.run( 16 * 1000 * 10 );
	sample.run( 16 * 1000 * 100 );
	sample.run( 16 * 1000 * 1000 );

	return EXIT_SUCCESS;
}
