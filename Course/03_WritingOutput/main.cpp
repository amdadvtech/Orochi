#include <common/Common.h>

class WritingOutputSample : public Sample
{
  public:
	void run( u32 size ) 
	{ 
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution( 0, 16 );

		Oro::GpuMemory<int> d_input( size );
		Oro::GpuMemory<int> d_output( size );
		Oro::GpuMemory<int> d_counters( 2 );

		std::vector<int> h_input( size );
		
		std::vector<const char*> opts;
		opts.push_back( "-I../" );

		Stopwatch sw;
		auto test = [&]( const char* kernelName )
		{
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../03_WritingOutput/Kernels.h", kernelName, &opts );
			const void* args[] = { &size, d_input.address(), d_output.address(), d_counters.address() };
			for( u32 i = 0; i < RunCount; ++i )
			{
				int h_counter = 0;
				for( u32 j = 0; j < size; ++j )
				{
					h_input[j] = distribution( generator );
					bool predicate = h_input[j] & 1;
					if( predicate ) ++h_counter;
				}
				d_input.copyFromHost( h_input.data(), size );

				OrochiUtils::memset( d_counters.ptr(), 0, 2 * sizeof( int ) );
				OrochiUtils::waitForCompletion();
				sw.start();

				OrochiUtils::launch1D( func, size, args, BlockSize );
				OrochiUtils::waitForCompletion();
				sw.stop();

				OROASSERT( h_counter == d_counters.getSingle(), 0 );

				float time = sw.getMs();
				float speed = static_cast<float>( size ) / 1000.0f / 1000.0f / time;
				float items = size / 1000.0f / 1000.0f;
				std::cout << std::setprecision( 2 ) << items << "M items output in " << time << " ms (" << speed << " GItems/s) [" << kernelName << "] " << std::endl;
			}
		};

		test( "WritingOutputNaiveKernel" );
		test( "WritingOutputKernel" );
		test( "WritingOutputBinaryKernel" );
		test( "WritingOutputComplementKernel" );
	}
};

int main( int argc, char** argv )
{
	WritingOutputSample sample;
	sample.run( 16 * 1000 * 1 );
	sample.run( 16 * 1000 * 10 );
	sample.run( 16 * 1000 * 100 );
	sample.run( 16 * 1000 * 1000 );

	return EXIT_SUCCESS;
}
