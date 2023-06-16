#include <common/Common.h>

class PrefixScanSample : public Sample
{
  public:
	void run( u32 size ) 
	{ 
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution( 0, 16 );

		// Input is an array of integers
		Oro::GpuMemory<int> d_input( size );
		// Output is an array of integers (prefix scan)
		Oro::GpuMemory<int> d_output( size );
		// Block counter for device-wise prefix scan
		Oro::GpuMemory<int> d_counter( 1 );
		// Offset sum for device-wise prefix scan
		Oro::GpuMemory<int> d_sum( 1 );

		std::vector<int> h_input( size );
		std::vector<int> h_output( size );
		
		std::vector<const char*> opts;
		opts.push_back( "-I../" );

		Stopwatch sw;
		auto test = [&]( const char* kernelName )
		{
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../02_PrefixScan/Kernels.h", kernelName, &opts );
			const void* args[] = { &size, d_input.address(), d_output.address(), d_sum.address(), d_counter.address() };
			for( u32 i = 0; i < RunCount; ++i )
			{
				h_output[0] = 0;
				for( u32 j = 0; j < size; ++j )
				{
					h_input[j] = distribution( generator );
					// Compute the correct result on CPU
					if( j == 0 )
						h_output[j] = h_input[j];
					else
						h_output[j] = h_input[j] + h_output[j - 1];
				}
				d_input.copyFromHost( h_input.data(), size );

				// Reset the block counter and the offset sum
				OrochiUtils::memset( d_counter.ptr(), 0, sizeof( int ) );
				OrochiUtils::memset( d_sum.ptr(), 0, sizeof( int ) );
				OrochiUtils::waitForCompletion();
				sw.start();

				OrochiUtils::launch1D( func, size, args, BlockSize, BlockSize * sizeof( int ) );
				OrochiUtils::waitForCompletion();
				sw.stop();

				// Validate the GPU result against the CPU one
				std::vector<int> output = d_output.getData();
				for( u32 j = 0; j < size; ++j )
					OROASSERT( h_output[j] == output[j], 0 );

				float time = sw.getMs();
				float speed = static_cast<float>( size ) / 1000.0f / 1000.0f / time;
				float items = size / 1000.0f / 1000.0f;
				std::cout << std::setprecision( 2 ) << items << "M items scanned in " << time << " ms (" << speed << " GItems/s) [" << kernelName << "] " << std::endl;
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
