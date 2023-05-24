#include <common/Common.h>

class WaterfallSchemeSample : public Sample
{
  public:
	void buildTree( const std::vector<int>& input, std::vector<Node>& nodes, std::vector<Leaf>& leaves ) 
	{ 
		Node root = { 0, input.size(), -1 };
		nodes[0] = root;
		int nodeCount = 1;

		std::stack<int> stack;
		stack.push( 0 );

		while( !stack.empty() )
		{
			int nodeIndex = stack.top();
			stack.pop();

			Node& node = nodes[nodeIndex];
			int l = node.m_left;
			int r = node.m_right;
			int m = ( l + r ) / 2;

			if( m - l > 1 )
			{
				int childIndex = nodeCount++;
				node.m_left = childIndex;
				nodes[childIndex] = { l, m, nodeIndex };
				stack.push( childIndex );
			}
			else
			{
				node.m_left = ~l;
				leaves[l] = { input[l], nodeIndex };
			}

			if( r - m > 1 )
			{
				int childIndex = nodeCount++;
				node.m_right = childIndex;
				nodes[childIndex] = { m, r, nodeIndex };
				stack.push( childIndex );
			}
			else
			{
				node.m_right = ~m;
				leaves[m] = { input[m], nodeIndex };
			}
		}
	}

	void run( u32 size ) 
	{ 
		assert( size > 1 );
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution( 0, 16 );

		Oro::GpuMemory<int> d_counters( 2 );
		Oro::GpuMemory<int> d_taskQueue( size - 1 );
		Oro::GpuMemory<Node> d_inNodes( size - 1 );
		Oro::GpuMemory<Node> d_outNodes( size - 1 );
		Oro::GpuMemory<Leaf> d_leaves( size );

		std::vector<int> h_input( size );
		std::vector<Node> h_nodes( size - 1);
		std::vector<Leaf> h_leaves( size );
		
		std::vector<const char*> opts;
		opts.push_back( "-I../" );
		opts.push_back( "-G" );

		Stopwatch sw;
		auto test = [&]( const char* kernelName )
		{
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../05_WaterfallScheme/Kernels.h", kernelName, &opts );
			const void* args[] = { &size, d_inNodes.address(), d_leaves.address(), d_taskQueue.address(), d_counters.address(), d_outNodes.address() };
			for( u32 i = 0; i < RunCount; ++i )
			{
				int h_sum = 0;
				for( u32 j = 0; j < size; ++j )
				{
					h_input[j] = distribution( generator );
					h_sum += h_input[j];
				}
				buildTree( h_input, h_nodes, h_leaves );
				d_inNodes.copyFromHost( h_nodes.data(), size - 1 );
				d_leaves.copyFromHost( h_leaves.data(), size );

				OrochiUtils::memset( d_counters.ptr(), 0, 2 * sizeof( int ) );
				OrochiUtils::memset( d_counters.ptr(), 1, sizeof( int ) );
				OrochiUtils::memset( d_taskQueue.ptr(), -1, ( size - 1 ) * sizeof( int ) );
				OrochiUtils::memset( d_taskQueue.ptr(), 0, sizeof( int ) );
				OrochiUtils::waitForCompletion();
				sw.start();

				OrochiUtils::launch1D( func, size, args, BlockSize );
				OrochiUtils::waitForCompletion();
				sw.stop();

				OROASSERT( d_counters.getSingle() == size - 1, 0 );

				float time = sw.getMs();
				float speed = static_cast<float>( size ) / 1000.0f / 1000.0f / time;
				float items = size / 1000.0f / 1000.0f;
				std::cout << "Tree with " << std::setprecision( 2 ) << items << "M items converted in " << time << " ms (" << speed << " GItems/s)  [" << kernelName << "]" << std::endl;
			}
		};

		test( "ConvertToBFSKernel" );
	}
};

int main( int argc, char** argv )
{
	WaterfallSchemeSample sample;
	sample.run( 16 );
	sample.run( 16 * 1000 * 1 );
	sample.run( 16 * 1000 * 10 );
	sample.run( 16 * 1000 * 100 );
	sample.run( 16 * 1000 * 1000 );

	return EXIT_SUCCESS;
}
