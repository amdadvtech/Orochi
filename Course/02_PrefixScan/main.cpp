#include <common/Common.h>

class PrefixScanSample : public Sample
{
  public:
	void run( u32 size ) 
	{ 
		// At least two items
		assert( size > 1 );

		// Random number generator
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution( 0, 16 );

		// Device input buffer
		Oro::GpuMemory<int> d_input( size );
		// Device output buffer
		Oro::GpuMemory<int> d_output( size );
		// Device block counter for device-wise prefix scan
		Oro::GpuMemory<int> d_counter( 1 );
		// Device offset sum for device-wise prefix scan
		Oro::GpuMemory<int> d_sum( 1 );

		// Host input and output buffers
		std::vector<int> h_input( size );
		std::vector<int> h_output( size );
		
		// Compiler options
		std::vector<const char*> opts;
		opts.push_back( "-I../" );

		Stopwatch sw;
		auto test = [&]( const char* kernelName )
		{
			// Compile function from the source code caching the compiled module
			// Sometimes it is necesarry to clear the cache (Course/build/cache)
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../02_PrefixScan/Kernels.h", kernelName, &opts );

			// Kernel arguments
			const void* args[] = { &size, d_input.address(), d_output.address(), d_sum.address(), d_counter.address() };
			
			// Run the kernel multiple times
			for( u32 i = 0; i < RunCount; ++i )
			{
				// Generate input values
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
				// Copy the input data to GPU
				d_input.copyFromHost( h_input.data(), size );

				// Reset the block counter and the offset sum
				OrochiUtils::memset( d_counter.ptr(), 0, sizeof( int ) );
				OrochiUtils::memset( d_sum.ptr(), 0, sizeof( int ) );

				// Synchronize before measuring
				OrochiUtils::waitForCompletion();
				sw.start();

				// Launch the kernel
				OrochiUtils::launch1D( func, size, args, BlockSize, BlockSize * sizeof( int ) );

				// Synchronize and stop measuring the executing time
				OrochiUtils::waitForCompletion();
				sw.stop();

				// Validate the results
				std::vector<int> output = d_output.getData();
				for( u32 j = 0; j < size; ++j )
					OROASSERT( h_output[j] == output[j], 0 );

				// Print the statistics
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
