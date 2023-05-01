#include <common/Common.h>
#include <common/BvhBuilder.h>

class WritingOutputSample : public Sample
{
  public:
	void run( u32 size ) 
	{ 
		constexpr int Bins = 128;
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution( 0, Bins - 1 );

		Oro::GpuMemory<int> d_input( size );
		Oro::GpuMemory<int> d_output( Bins );
		Oro::GpuMemory<int> d_counter( 1 );

		std::vector<int> h_input( size );
		
		std::vector<const char*> opts;
		opts.push_back( "-I../" );

		oroDeviceProp prop;
		CHECK_ORO( oroGetDeviceProperties( &prop, m_device ) );
		u32 warpSize = prop.gcnArch >= 1010 || std::string( prop.name ).find( "NVIDIA" ) != std::string::npos ? 32u : 64u;

		Stopwatch sw;
		auto test = [&]( const char* kernelName )
		{
			// TODO: use the exact formula
			u32 threads = prop.multiProcessorCount * warpSize;
			oroFunction func = m_utils.getFunctionFromFile( m_device, "../06_PersistentThreads/Kernels.h", kernelName, &opts );
			const void* args[] = { &size, &threads, &Bins, d_input.address(), d_output.address(), d_counter.address() };
			for( u32 i = 0; i < RunCount; ++i )
			{
				std::vector<int> h_output( Bins );
				for( u32 j = 0; j < size; ++j )
				{
					h_input[j] = distribution( generator );
					h_output[h_input[j]]++;
				}
				d_input.copyFromHost( h_input.data(), size );

				OrochiUtils::memset( d_counter.ptr(), 0, sizeof( int ) );
				OrochiUtils::waitForCompletion();
				sw.start();

				OrochiUtils::launch1D( func, threads, args, 32 );
				OrochiUtils::waitForCompletion();
				sw.stop();

				std::vector<int> output = d_output.getData();
				for( u32 j = 0; j < Bins; ++j )
					OROASSERT( h_output[j] == output[j], 0 );

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
	WritingOutputSample sample;
	sample.run( 16 * 1000 * 1 );
	sample.run( 16 * 1000 * 10 );
	sample.run( 16 * 1000 * 100 );
	sample.run( 16 * 1000 * 1000 );

	return EXIT_SUCCESS;
}
