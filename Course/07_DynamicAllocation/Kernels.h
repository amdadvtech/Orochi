#include <common/Common.h>

class Stack
{
  public:
	__device__ Stack( u32 stackSize, u32 stackCount, int* stackBuffer, int* locks )
	{
		u32 threadIndex = threadIdx.x + threadIdx.y * blockDim.x;
		u32 warpIndex = threadIndex / warpSize;
		u32 laneIndex = threadIndex & ( warpSize - 1 );

		u32 warpsPerBlock = ( blockDim.x * blockDim.y + warpSize - 1 ) / warpSize;
		u32 activeWarps = stackCount / warpSize;
		u32 firstThreadIndex = __ffsll( __ballot( true ) ) - 1;

		u32 warpHash = 0xffffffff;
		u32 warpHashCandidate = ( warpIndex + ( blockIdx.x + blockIdx.y * gridDim.x ) * warpsPerBlock ) % activeWarps;
		while( warpHash == 0xffffffff )
		{
			if( laneIndex == firstThreadIndex )
			{
				if( atomicCAS( &locks[warpHashCandidate], 0, 1 ) == 0 ) warpHash = warpHashCandidate;
			}
			warpHashCandidate = ( warpHashCandidate + 1 ) % activeWarps;
			warpHash = __shfl( warpHash, firstThreadIndex );
		}
		m_lock = &locks[warpHash];

		u32 offset = activeWarps + laneIndex + ( warpHash * warpSize ) * stackSize;
		m_buffer = stackBuffer + offset;
		m_size = stackSize;
	}

	__device__ ~Stack()
	{
		u32 threadIndex = threadIdx.x + threadIdx.y * blockDim.x;
		u32 laneIndex = threadIndex & ( warpSize - 1 );
		u32 firstThreadIndex = __ffsll( __ballot( true ) ) - 1;
		if( laneIndex == firstThreadIndex ) atomicExch( m_lock, 0 );
	}

	__device__ int pop() { return m_buffer[--m_index * warpSize]; }
	__device__ void push( int val ) { m_buffer[m_index++ * warpSize] = val; }
	__device__ bool empty() const { return m_index == 0; }

  private:
	u32 m_size;
	int m_index = 0;
	int* m_lock;
	int* m_buffer;
};

extern "C" __global__ void CountKernel( u32 size, u32 stackSize, u32 stackCount, int* stackBuffer, int* locks, const Node* nodes, const Leaf* leaves, int* queries, int* counts )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index >= size ) return;

	int query = queries[index];
	int count = 0;

	Stack stack( stackSize, stackCount, stackBuffer, locks );
	stack.push( 0 );
	while( !stack.empty() )
	{
		int nodeIndex = stack.pop();
		if( !isLeaf( nodeIndex ) )
		{
			const Node& node = nodes[nodeIndex];
			if( node.m_pivot <= query ) stack.push( node.m_rightIndex );
			if( node.m_pivot >= query ) stack.push( node.m_leftIndex );
		}
		else
		{
			const Leaf& leaf = leaves[~nodeIndex];
			if( leaf.m_value == query ) ++count;
		}
	}
	counts[index] = count;
}
