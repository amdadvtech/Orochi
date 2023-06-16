#include <common/Common.h>

class DynamicAllocationSample : public Sample
{
  public:
	void run( u32 size ) 
	{ 
		// At least two items
		assert( size > 1 );

		// Random number generator
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution;
		
		// Query device properties
		oroDeviceProp prop;
		CHECK_ORO( oroGetDeviceProperties( &prop, m_device ) );

		// Determine how many stack to allocate (for active threads)
		//u32 stackCount = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
		u32 stackCount = prop.multiProcessorCount * 64;
		
		// The stack size
		u32 stackSize = 128u;

		// Device stack buffer (for all stacks)
		Oro::GpuMemory<int> d_stackBuffer( stackSize  * stackCount );
		// Device locks for allocations
		Oro::GpuMemory<int> d_locks( stackCount );
		// Device output counts
		Oro::GpuMemory<int> d_counts( size );
		// Device input queries
		Oro::GpuMemory<int> d_queries( size );
		// Device input internal nodes of the binary tree
		Oro::GpuMemory<Node> d_nodes( size - 1 );
		// Device input leaves of the binary tree
		Oro::GpuMemory<Leaf> d_leaves( size );

		// Host input buffer
		std::vector<int> h_input( size );
		// Host input queries
		std::vector<int> h_queries( size );
		// Host internal nodes of the binary tree
		std::vector<Node> h_nodes( size - 1 );
		// Host leaves of the binary tree
		std::vector<Leaf> h_leaves( size );

		std::vector<const char*> opts;
		opts.push_back( "-I../" );

		// Compiler options
		Stopwatch sw;
		auto test = [&]( const char* kernelName )
		{
			// Compile function from the source code caching the compiled module
			// Sometimes it is necesarry to clear the cache (Course/build/cache)
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../07_DynamicAllocation/Kernels.h", kernelName, &opts );

			// Kernel arguments
			const void* args[] = { &size, &stackSize, &stackCount, d_stackBuffer.address(), d_locks.address(), d_nodes.address(), d_leaves.address(), d_queries.address(), d_counts.address() };

			// Run the kernel multiple times
			for( u32 i = 0; i < RunCount; ++i )
			{
				// Generate input and queries values
				for( u32 j = 0; j < size; ++j )
				{
					h_queries[j] = distribution( generator );
					h_input[j] = distribution( generator );
				}

				// Build the tree and copy it to GPU
				TreeBuilder().build( h_input, h_nodes, h_leaves );
				d_nodes.copyFromHost( h_nodes.data(), size - 1 );
				d_leaves.copyFromHost( h_leaves.data(), size );
				d_queries.copyFromHost( h_queries.data(), size );

				OrochiUtils::memset( d_locks.ptr(), 0, stackCount * sizeof( int ) );

				// Synchronize before measuring
				OrochiUtils::waitForCompletion();
				sw.start();

				// Launch the kernel
				OrochiUtils::launch1D( func, size, args, BlockSize );

				// Synchronize and stop measuring the executing time
				OrochiUtils::waitForCompletion();
				sw.stop();

				int h_count = 0;
				for( u32 j = 0; j < size; ++j )
					if( h_queries[0] == h_input[j] ) ++h_count;
				OROASSERT( h_count == d_counts.getSingle(), 0 );

				// Print the statistics
				float time = sw.getMs();
				float speed = static_cast<float>( size ) / 1000.0f / 1000.0f / time;
				float items = size / 1000.0f / 1000.0f;
				std::cout << std::setprecision( 2 ) << items << "M items counted in " << time << " ms (" << speed << " GItems/s)  [" << kernelName << "]" << std::endl;
			}
		};

		test( "CountKernel" );
	}
};

int main( int argc, char** argv )
{
	DynamicAllocationSample sample;
	sample.run( 16 * 1000 * 1 );
	sample.run( 16 * 1000 * 10 );
	sample.run( 16 * 1000 * 100 );
	sample.run( 16 * 1000 * 1000 );

	return EXIT_SUCCESS;
}
