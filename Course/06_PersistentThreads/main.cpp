#include <common/Common.h>

class PersistentThreadsSample : public Sample
{
  public:
	void run( u32 size ) 
	{ 
		// At least two items
		assert( size > 1 );

		// The number of histogram bins
		constexpr int Bins = 128;

		// Random number generator
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution( 0, Bins - 1 );

		// Device input buffer
		Oro::GpuMemory<int> d_input( size );
		// Device histogram
		Oro::GpuMemory<int> d_output( Bins );
		// Device counter for the global barrier
		Oro::GpuMemory<int> d_counter( 1 );

		// Host input buffer
		std::vector<int> h_input( size );
		
		// Compiler options
		std::vector<const char*> opts;
		opts.push_back( "-I../" );

		// Query device properties
		oroDeviceProp prop;
		CHECK_ORO( oroGetDeviceProperties( &prop, m_device ) );

		Stopwatch sw;
		auto test = [&]( const char* kernelName )
		{
			// Compile function from the source code caching the compiled module
			// Sometimes it is necesarry to clear the cache (Course/build/cache)
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../06_PersistentThreads/Kernels.h", kernelName, &opts );

			// Query the number of active blocks per SM via the occupancy API
			int blockCount;
			CHECK_ORO( oroDrvOccupancyMaxActiveBlocksPerMultiprocessor( &blockCount, func, BlockSize, 0 ) );

			// The number of persistent threads
			u32 threads = prop.multiProcessorCount * blockCount * prop.warpSize;

			// Kernel arguments
			const void* args[] = { &size, &threads, &Bins, d_input.address(), d_output.address(), d_counter.address() };
			
			// Run the kernel multiple times
			for( u32 i = 0; i < RunCount; ++i )
			{
				// Generate input values
				std::vector<int> h_output( Bins );
				for( u32 j = 0; j < size; ++j )
				{
					h_input[j] = distribution( generator );
					h_output[h_input[j]]++;
				}
				// Copy the input data to GPU
				d_input.copyFromHost( h_input.data(), size );

				// Reset the counter for the global barrier
				OrochiUtils::memset( d_counter.ptr(), 0, sizeof( int ) );

				// Synchronize before measuring
				OrochiUtils::waitForCompletion();
				sw.start();

				// Launch the kernel
				OrochiUtils::launch1D( func, threads, args, BlockSize );

				// Synchronize and stop measuring the executing time
				OrochiUtils::waitForCompletion();
				sw.stop();

				std::vector<int> output = d_output.getData();
				for( u32 j = 0; j < Bins; ++j )
					OROASSERT( h_output[j] == output[j], 0 );

				// Print the statistics
				float time = sw.getMs();
				float speed = static_cast<float>( size ) / 1000.0f / 1000.0f / time;
				float items = size / 1000.0f / 1000.0f;
				std::cout << "Histogram of " << std::setprecision( 2 ) << items << " computed in " << time << " ms (" << speed << " GItems/s) [" << kernelName << "] " << std::endl;
			}
		};

		test( "HistogramKernel" );
	}
};

int main( int argc, char** argv )
{
	PersistentThreadsSample sample;
	sample.run( 16 * 1000 * 1 );
	sample.run( 16 * 1000 * 10 );
	sample.run( 16 * 1000 * 100 );
	sample.run( 16 * 1000 * 1000 );

	return EXIT_SUCCESS;
}
