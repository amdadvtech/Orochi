#include <common/Common.h>

class WaterfallSchemeSample : public Sample
{
  public:
	void run( u32 size ) 
	{ 
		assert( size > 1 );
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution( 0, 16 );

		// Device input buffer
		Oro::GpuMemory<int> d_input( size );
		// Two counters for counting leaves and internal nodes
		Oro::GpuMemory<int> d_counters( 2 );
		// Task queue
		Oro::GpuMemory<int> d_taskQueue( size - 1 );
		// Device output internal nodes of the binary tree
		Oro::GpuMemory<Node> d_nodes( size - 1 );
		// Device Output leaves of the binary tree
		Oro::GpuMemory<Leaf> d_leaves( size );

		// Host input buffer
		std::vector<int> h_input( size );
		// Host internal nodes of the binary tree
		std::vector<Node> h_nodes( size - 1);
		// Host leaves of the binary tree
		std::vector<Leaf> h_leaves( size );
		
		// Compiler options
		std::vector<const char*> opts;
		opts.push_back( "-I../" );

		Stopwatch sw;
		auto test = [&]( const char* kernelName )
		{
			// Compile function from the source code caching the compiled module
			// Sometimes it is necesarry to clear the cache (Course/build/cache)
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../05_WaterfallScheme/Kernels.h", kernelName, &opts );
			
			// Kernel arguments
			const void* args[] = { &size, d_input.address(), d_taskQueue.address(), d_counters.address(), d_nodes.address(), d_leaves.address() };
			
			// Run the kernel multiple times
			for( u32 i = 0; i < RunCount; ++i )
			{
				// Generate input values
				int h_sum = 0;
				for( u32 j = 0; j < size; ++j )
				{
					h_input[j] = distribution( generator );
					h_sum += h_input[j];
				}
				// Copy the input data to GPU
				d_input.copyFromHost( h_input.data(), size );

				// Build the tree and copy it to GPU
				TreeBuilder().build( h_input, h_nodes, h_leaves );

				// Reset both counters
				OrochiUtils::memset( d_counters.ptr(), 0, 2 * sizeof( int ) );

				// The first counter is used for internal nodes
				// We already have offset for root, which is placed at position 0
				// For other nodes, it starts from 1, and that's we set it to 1
				OrochiUtils::memset( d_counters.ptr(), 1, sizeof( int ) );

				// Copy root node to GPU
				Node root = { 0, size, -1 };
				OrochiUtils::copyHtoD( d_nodes.ptr(), &root, 1 );

				// Set all tasks to 0 except the first one
				// The first task correspond to the root
				OrochiUtils::memset( d_taskQueue.ptr(), 0, ( size - 1 ) * sizeof( int ) );
				OrochiUtils::memset( d_taskQueue.ptr(), 1, sizeof( int ) );

				// Synchronize before measuring
				OrochiUtils::waitForCompletion();
				sw.start();

				// Launch the kernel
				OrochiUtils::launch1D( func, size, args, BlockSize );

				// Synchronize and stop measuring the executing time
				OrochiUtils::waitForCompletion();
				sw.stop();

				// Check the number of internal nodes processed
				OROASSERT( d_counters.getSingle() == size - 1, 0 );

				// Print the statistics
				float time = sw.getMs();
				float speed = static_cast<float>( size ) / 1000.0f / 1000.0f / time;
				float items = size / 1000.0f / 1000.0f;
				std::cout << "Tree with " << std::setprecision( 2 ) << items << "M items constructed in " << time << " ms (" << speed << " GItems/s)  [" << kernelName << "]" << std::endl;
			}
		};

		test( "BuildTree" );
	}
};

int main( int argc, char** argv )
{
	WaterfallSchemeSample sample;
	sample.run( 16 * 1000 * 1 );
	sample.run( 16 * 1000 * 10 );
	sample.run( 16 * 1000 * 100 );
	sample.run( 16 * 1000 * 1000 );

	return EXIT_SUCCESS;
}
