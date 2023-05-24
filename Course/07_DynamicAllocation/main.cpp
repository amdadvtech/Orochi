#include <common/Common.h>

class DynamicAllocationSample : public Sample
{
  public:
	void run( u32 size ) 
	{ 
		assert( size > 1 );
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution( 0, 16 );
		
		oroDeviceProp prop;
		CHECK_ORO( oroGetDeviceProperties( &prop, m_device ) );
		//u32 stackCount = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
		u32 stackCount = prop.multiProcessorCount * 64;
		u32 stackSize = 128u;

		Oro::GpuMemory<int> d_stackBuffer( stackSize  * stackCount );
		Oro::GpuMemory<int> d_locks( stackCount );
		Oro::GpuMemory<int> d_counts( size );
		Oro::GpuMemory<int> d_queries( size );
		Oro::GpuMemory<Node> d_nodes( size - 1 );
		Oro::GpuMemory<Leaf> d_leaves( size );

		std::vector<int> h_input( size );
		std::vector<int> h_queries( size );
		std::vector<Node> h_nodes( size - 1 );
		std::vector<Leaf> h_leaves( size );

		std::vector<const char*> opts;
		opts.push_back( "-I../" );
		opts.push_back( "-G");

		Stopwatch sw;
		auto test = [&]( const char* kernelName )
		{
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../07_DynamicAllocation/Kernels.h", kernelName, &opts );
			const void* args[] = { &size, &stackSize, &stackCount, d_stackBuffer.address(), d_locks.address(), d_nodes.address(), d_leaves.address(), d_queries.address(), d_counts.address() };
			for( u32 i = 0; i < RunCount; ++i )
			{
				for( u32 j = 0; j < size; ++j )
				{
					h_queries[j] = distribution( generator );
					h_input[j] = distribution( generator );
				}
				TreeBuilder().build( h_input, h_nodes, h_leaves );
				d_nodes.copyFromHost( h_nodes.data(), size - 1 );
				d_leaves.copyFromHost( h_leaves.data(), size );
				d_queries.copyFromHost( h_queries.data(), size );

				OrochiUtils::memset( d_locks.ptr(), 0, stackCount * sizeof( int ) );
				OrochiUtils::waitForCompletion();
				sw.start();

				OrochiUtils::launch1D( func, size, args, BlockSize );
				OrochiUtils::waitForCompletion();
				sw.stop();

				int h_count = 0;
				for( u32 j = 0; j < size; ++j )
					if( h_queries[0] == h_input[j] ) ++h_count;
				//OROASSERT( h_count == d_counts.getSingle(), 0 );

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
