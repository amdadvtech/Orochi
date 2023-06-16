#include <common/Common.h>

class BottomUpTraversalSample : public Sample
{
  public:
	void run( u32 size ) 
	{ 
		assert( size > 1 );
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution( 0, 16 );

		// Counters for internal nodes
		Oro::GpuMemory<int> d_counters( size - 1 );
		// Output array with sums for internal nodes
		Oro::GpuMemory<int> d_sums( size - 1 );
		// Input binary tree - internal nodes
		Oro::GpuMemory<Node> d_nodes( size - 1 );
		// Input binary tree - leaves
		Oro::GpuMemory<Leaf> d_leaves( size );

		std::vector<int> h_input( size );
		std::vector<Node> h_nodes( size - 1);
		std::vector<Leaf> h_leaves( size );
		
		std::vector<const char*> opts;
		opts.push_back( "-I../" );

		Stopwatch sw;
		auto test = [&]( const char* kernelName )
		{
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../04_BottomUpTraversal/Kernels.h", kernelName, &opts );
			const void* args[] = { &size, d_nodes.address(), d_leaves.address(), d_sums.address(), d_counters.address() };
			for( u32 i = 0; i < RunCount; ++i )
			{
				int h_sum = 0;
				for( u32 j = 0; j < size; ++j )
				{
					h_input[j] = distribution( generator );
					// Summing up all elements on CPU
					h_sum += h_input[j];
				}
				TreeBuilder().build( h_input, h_nodes, h_leaves );
				d_nodes.copyFromHost( h_nodes.data(), size - 1 );
				d_leaves.copyFromHost( h_leaves.data(), size );

				// Reset counters
				OrochiUtils::memset( d_counters.ptr(), 0, ( size - 1 ) * sizeof( int ) );
				OrochiUtils::waitForCompletion();
				sw.start();

				OrochiUtils::launch1D( func, size, args, BlockSize );
				OrochiUtils::waitForCompletion();
				sw.stop();

				// Validate the sum in the root
				OROASSERT( h_sum == d_sums.getSingle(), 0 );

				float time = sw.getMs();
				float speed = static_cast<float>( size ) / 1000.0f / 1000.0f / time;
				float items = size / 1000.0f / 1000.0f;
				std::cout << std::setprecision( 2 ) << items << "M items summed in " << time << " ms (" << speed << " GItems/s)  [" << kernelName << "]" << std::endl;
			}
		};

		test( "BottomUpTraversalKernel" );
	}
};

int main( int argc, char** argv )
{
	BottomUpTraversalSample sample;
	sample.run( 16 * 1000 * 1 );
	sample.run( 16 * 1000 * 10 );
	sample.run( 16 * 1000 * 100 );
	sample.run( 16 * 1000 * 1000 );

	return EXIT_SUCCESS;
}
