#include <common/Common.h>

class ReductionSample : public Sample
{
  public:
	void run( u32 size ) 
	{ 
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution( 0, 16 );

		// Input is an array of integers
		Oro::GpuMemory<int> d_input( size );
		// Output is a single interger
		Oro::GpuMemory<int> d_output( 1 );

		std::vector<int> h_input( size );

		std::vector<const char*> opts;
		opts.push_back( "-I../" );

		Stopwatch sw;
		auto test = [&]( const char* kernelName )
		{
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../01_Reduction/Kernels.h", kernelName, &opts );
			const void* args[] = { &size, d_input.address(), d_output.address() };
			for( u32 i = 0; i < RunCount; ++i )
			{
				int h_output = 0;
				for( u32 j = 0; j < size; ++j )
				{
					h_input[j] = distribution( generator );
					// Compute the correct result on CPU
					h_output += h_input[j];
				}
				d_input.copyFromHost( h_input.data(), size );

				// Reset the global counter
				OrochiUtils::memset( d_output.ptr(), 0, sizeof( int ) );
				OrochiUtils::waitForCompletion();
				sw.start();

				OrochiUtils::launch1D( func, size, args, BlockSize, BlockSize * sizeof( int ) );
				OrochiUtils::waitForCompletion();
				sw.stop();

				// Validate the GPU result
				OROASSERT( h_output == d_output.getSingle(), 0 );

				float time = sw.getMs();
				float speed = static_cast<float>( size ) / 1000.0f / 1000.0f / time;
				float items = size / 1000.0f / 1000.0f;
				std::cout << std::setprecision( 2 ) << items << "M items reduced in " << time << " ms (" << speed << " GItems/s)  [" << kernelName << "]" << std::endl;
			}
		};

		test( "ReduceBlockKernel" );
		test( "ReduceWarpKernel" );
	}
};

int main( int argc, char** argv )
{
	ReductionSample sample;
	sample.run( 16 * 1000 * 1 );
	sample.run( 16 * 1000 * 10 );
	sample.run( 16 * 1000 * 100 );
	sample.run( 16 * 1000 * 1000 );

	return EXIT_SUCCESS;
}
