#include <common/Common.h>

class EnqueueSample : public Sample
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
		// Device counters
		Oro::GpuMemory<int> d_counters( 2 );

		// Host input buffer
		std::vector<int> h_input( size );
		
		// Compiler options
		std::vector<const char*> opts;
		opts.push_back( "-I../" );

		Stopwatch sw;
		auto test = [&]( const char* kernelName, u32 threadCount )
		{
			// Compile function from the source code caching the compiled module
			// Sometimes it is necesarry to clear the cache (Course/build/cache)
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../03_Enqueue/Kernels.h", kernelName, &opts );

			// Kernel arguments
			const void* args[] = { &size, d_input.address(), d_output.address(), d_counters.address() };

			// Run the kernel multiple times
			for( u32 i = 0; i < RunCount; ++i )
			{
				// Generate input values
				int h_counter = 0;
				for( u32 j = 0; j < size; ++j )
				{
					h_input[j] = distribution( generator );
					bool enqueue = h_input[j] & 1;
					// We just compute the number of elements satisfying the predicate
					if( enqueue ) ++h_counter;
				}
				// Copy the input data to GPU
				d_input.copyFromHost( h_input.data(), size );

				// Reset global counters
				OrochiUtils::memset( d_counters.ptr(), 0, 2 * sizeof( int ) );

				// Synchronize before measuring
				OrochiUtils::waitForCompletion();
				sw.start();

				// Launch the kernel
				OrochiUtils::launch1D( func, size, args, BlockSize );

				// Synchronize and stop measuring the executing time
				OrochiUtils::waitForCompletion();
				sw.stop();

				// Validate the result
				// We just check the number of enqueued elements
				OROASSERT( h_counter == d_counters.getSingle(), 0 );

				// Print the statistics
				float time = sw.getMs();
				float speed = static_cast<float>( size ) / 1000.0f / 1000.0f / time;
				float items = size / 1000.0f / 1000.0f;
				std::cout << std::setprecision( 2 ) << items << "M items output in " << time << " ms (" << speed << " GItems/s) [" << kernelName << "] " << std::endl;
			}
		};

		test( "EnqueueNaiveKernel", size );
		test( "EnqueueKernel", size / 2 );
		test( "EnqueueBinaryKernel", size );
		test( "EnqueueComplementKernel", size );
	}
};

int main( int argc, char** argv )
{
	EnqueueSample sample;
	sample.run( 16 * 1000 * 1 );
	sample.run( 16 * 1000 * 10 );
	sample.run( 16 * 1000 * 100 );
	sample.run( 16 * 1000 * 1000 );

	return EXIT_SUCCESS;
}
