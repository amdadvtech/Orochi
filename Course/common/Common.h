#pragma once

#if( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define __KERNELCC__
#endif

#ifndef __KERNELCC__
#include <Orochi/Orochi.h>
#include <Orochi/OrochiUtils.h>
#include <Orochi/GpuMemory.h>
#include <Test/Stopwatch.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <stack>
#include <vector>
#include <random>
#endif

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

struct alignas( 32 ) Node
{
	int m_left;
	int m_right;
	int m_parent;
};

struct alignas( 8 ) Leaf
{
	int m_value;
	int m_parent;
};

#ifndef __KERNELCC__
#define CHECK_ORO( error ) ( checkOro( error, __FILE__, __LINE__ ) )
void checkOro( oroError res, const char* file, int line );

#define CHECK_ORORTC( error ) ( checkOrortc( error, __FILE__, __LINE__ ) )
void checkOrortc( orortcResult res, const char* file, int line );

class Sample
{
  public:
	static constexpr int DeviceIndex = 0;
	static constexpr int RunCount = 4;
	static constexpr int BlockSize = 1024;

	Sample()
	{
		oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0 );

		CHECK_ORO( oroInit( 0 ) );
		CHECK_ORO( oroDeviceGet( &m_device, DeviceIndex ) );
		CHECK_ORO( oroCtxCreate( &m_context, 0, m_device ) );

		oroDeviceProp props;
		CHECK_ORO( oroGetDeviceProperties( &props, m_device ) );
		std::cout << "Executing on '" << props.name << "'" << std::endl;
	}

	~Sample() { CHECK_ORO( oroCtxDestroy( m_context ) ); }

  protected:
	oroCtx m_context;
	oroDevice m_device;
	OrochiUtils m_utils;
};
#endif
