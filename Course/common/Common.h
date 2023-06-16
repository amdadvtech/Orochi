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

typedef long long int i64;

#ifndef __KERNELCC__
#define __device__
#endif

struct alignas( 32 ) Node
{
	int m_leftIndex;
	int m_rightIndex;
	int m_parentAddr;
	int m_pivot;

	__device__ bool isLeftLeaf() const { return m_leftIndex < 0; }
	__device__ bool isRightLeaf() const { return m_rightIndex < 0; }

	__device__ int getLeftAddr() const { return m_leftIndex < 0 ? ~m_leftIndex : m_leftIndex; }
	__device__ int getRightAddr() const { return m_rightIndex < 0 ? ~m_rightIndex : m_rightIndex; }

	__device__ void setLeftIndex( int leftIndex ) { m_leftIndex = leftIndex; }
	__device__ void setRightIndex( int rightIndex ) { m_rightIndex = rightIndex; }
};

struct alignas( 8 ) Leaf
{
	int m_value;
	int m_parentAddr;
};

__device__ static bool isLeaf( int nodeIndex ) { return nodeIndex < 0; }

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

	Sample( const Sample& ) = delete;
	Sample& operator=( const Sample& other ) = delete;

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

	virtual ~Sample() { CHECK_ORO( oroCtxDestroy( m_context ) ); }

  protected:
	oroCtx m_context;
	oroDevice m_device;
	OrochiUtils m_utils;
};

class TreeBuilder
{
  public:
	void build( std::vector<int>& input, std::vector<Node>& nodes, std::vector<Leaf>& leaves )
	{
		std::sort( input.begin(), input.end() );

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
			int l = node.m_leftIndex;
			int r = node.m_rightIndex;
			int m = ( l + r ) / 2;
			node.m_pivot = input[m];

			if( m - l > 1 )
			{
				int childIndex = nodeCount++;
				node.m_leftIndex = childIndex;
				nodes[childIndex] = { l, m, nodeIndex };
				stack.push( childIndex );
			}
			else
			{
				node.m_leftIndex = ~l;
				leaves[l] = { input[l], nodeIndex };
			}

			if( r - m > 1 )
			{
				int childIndex = nodeCount++;
				node.m_rightIndex = childIndex;
				nodes[childIndex] = { m, r, nodeIndex };
				stack.push( childIndex );
			}
			else
			{
				node.m_rightIndex = ~m;
				leaves[m] = { input[m], nodeIndex };
			}
		}
	}
};
#endif
