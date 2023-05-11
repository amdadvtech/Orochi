#include <common/Common.h>

class BottomUpTraversalSample : public Sample
{
  public:
	void buildTree( const std::vector<int>& input, std::vector<Node>& nodes, std::vector<Leaf>& leaves ) 
	{ 
		Node root = { 0, input.size(), -1, 0, 0 };
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
				nodes[childIndex] = { l, m, nodeIndex, 0, 0 };
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
				nodes[childIndex] = { m, r, nodeIndex, 0, 0 };
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

		Oro::GpuMemory<Node> d_nodes( size - 1 );
		Oro::GpuMemory<Leaf> d_leaves( size );

		std::vector<int> h_input( size );
		std::vector<Node> h_nodes( size - 1);
		std::vector<Leaf> h_leaves( size );
		
		std::vector<const char*> opts;
		opts.push_back( "-I../" );

		Stopwatch sw;
		auto test = [&]( const char* kernelName )
		{
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../05_BottomUpTraversal/Kernels.h", kernelName, &opts );
			const void* args[] = { &size, d_nodes.address(), d_leaves.address() };
			for( u32 i = 0; i < RunCount; ++i )
			{
				int h_sum = 0;
				for( u32 j = 0; j < size; ++j )
				{
					h_input[j] = distribution( generator );
					h_sum += h_input[j];
				}
				buildTree( h_input, h_nodes, h_leaves );
				d_nodes.copyFromHost( h_nodes.data(), size - 1 );
				d_leaves.copyFromHost( h_leaves.data(), size );

				OrochiUtils::waitForCompletion();
				sw.start();

				OrochiUtils::launch1D( func, size, args, BlockSize );
				OrochiUtils::waitForCompletion();
				sw.stop();

				OROASSERT( h_sum == d_nodes.getSingle().m_sum, 0 );

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
