

#include "contrib/hipew/include/hipew.h"
#include <Orochi/GpuMemory.h>
#include <Orochi/Orochi.h>
#include <Test/Common.h>
#include <iostream>

int main()
{
	if( oroInitialize( ORO_API_HIP, 0 ) != 0 )
	{
		std::cerr << "Unable to initialize Orochi. Please check your HIP installation or create an issue at our github for assistance.\n";
		return -1;
	}

	oroError e{};
	oroDevice device{};
	// Get the device at index 0
	e = oroDeviceGet( &device, 0 );
	ERROR_CHECK( e );

	static constexpr auto name_size = 128;
	char name[128];
	e = oroDeviceGetName( name, name_size, device );
	ERROR_CHECK( e );

	oroDeviceProp props{};
	e = oroGetDeviceProperties( &props, device );
	ERROR_CHECK( e );
	printf( "executing on %s (%s)\n", props.name, props.gcnArchName );

	oroCtx ctx{};
	e = oroCtxCreate( &ctx, 0, device );
	ERROR_CHECK( e );
	oroCtxSetCurrent( ctx );

	std::vector<char> code;
	const char* funcName = "texture_test";

	static constexpr auto filepath = "../Test/Texture/texture_test_kernel.hpp";

	load( filepath, code );

	orortcProgram prog;
	orortcResult rtc_e;
	rtc_e = orortcCreateProgram( &prog, code.data(), funcName, 0, 0, 0 );

	if( rtc_e != ORORTC_SUCCESS )
	{
		std::cerr << "orortcCreateProgram failed" << std::endl;
		return -1;
	}

	std::vector<const char*> opts;
	opts.push_back( "-I ../" );
	rtc_e = orortcCompileProgram( prog, opts.size(), opts.data() );
	if( rtc_e != ORORTC_SUCCESS )
	{
		std::cerr << "orortcCompileProgram failed" << std::endl;
		return -1;
	}

	size_t codeSize;
	rtc_e = orortcGetCodeSize( prog, &codeSize );

	std::vector<char> codec( codeSize );
	rtc_e = orortcGetCode( prog, codec.data() );
	rtc_e = orortcDestroyProgram( &prog );

	oroModule module;
	oroFunction function;
	e = oroModuleLoadData( &module, codec.data() );
	e = oroModuleGetFunction( &function, module, funcName );
	ERROR_CHECK( e );

	static constexpr auto grid_resolution = 256;
	static constexpr auto num_features = 4;

	Oro::GpuMemory<float> grid_data( grid_resolution * grid_resolution * num_features );

	hipArray_Format format = HIP_AD_FORMAT_FLOAT;

	// Resource Desc
	hipResourceDesc resDesc;
	std::memset( &resDesc, 0, sizeof( resDesc ) );

	resDesc.resType = hipResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = reinterpret_cast<hipDeviceptr_t>( grid_data.ptr() );

	resDesc.res.pitch2D.format = format;
	resDesc.res.pitch2D.numChannels = num_features;

	resDesc.res.pitch2D.width = grid_resolution;
	resDesc.res.pitch2D.height = grid_resolution;
	resDesc.res.pitch2D.pitchInBytes = grid_resolution * sizeof( float ) * num_features;

	hipTextureAddressMode address_mode = hipAddressModeWrap;
	hipTextureFilterMode filter_mode = hipFilterModeLinear;

	hipTextureDesc texDesc;
	std::memset( &texDesc, 0, sizeof( texDesc ) );
	texDesc.addressMode[0] = address_mode;
	texDesc.addressMode[1] = address_mode;
	texDesc.addressMode[2] = address_mode;
	texDesc.filterMode = filter_mode;
	texDesc.flags = HIP_TRSF_READ_AS_INTEGER;

	hipTextureObject_t textureObject{};
	const auto ret = hipTexObjectCreate( &textureObject, &resDesc, &texDesc, nullptr );

	if( ret == hipSuccess )
	{
		std::cerr << "hipTexObjectCreate succeed !" << std::endl;
	}
	else
	{
		std::cerr << "hipTexObjectCreate failed !" << std::endl;
	}

	oroStream stream;
	e = oroStreamCreate( &stream );
	ERROR_CHECK( e );

	static constexpr auto num_iters = 1;
	for( int i = 0; i < num_iters; ++i )
	{
		const void* args[] = { &textureObject };
		e = oroModuleLaunchKernel( function, 1, 1, 1, 32, 1, 1, 0, stream, (void**)args, nullptr );
		ERROR_CHECK( e );
	}

	e = oroStreamDestroy( stream );
	ERROR_CHECK( e );

	e = oroModuleUnload( module );
	ERROR_CHECK( e );

	e = oroCtxDestroy( ctx );
	ERROR_CHECK( e );

	return 0;
}