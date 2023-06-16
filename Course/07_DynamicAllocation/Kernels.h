#include <common/Common.h>

// Stack class
class Stack
{
  public:
	// A constructor dynamically allocating the stack buffer
	__device__ Stack( u32 stackSize, u32 stackCount, int* stackBuffer, int* locks )
	{
		// The number of warps per block
		u32 warpsPerBlock = ( blockDim.x + warpSize - 1 ) / warpSize;
		// Global warp index
		u32 warpIndex = threadIdx.x / warpSize + blockIdx.x * warpsPerBlock;
		
		// Lane index
		u32 laneIndex = threadIdx.x & ( warpSize - 1 );
		// First active lane in the warp
		u32 firstLaneIndex = __ffsll( __ballot( true ) ) - 1;
		
		// The number of active warps
		u32 activeWarps = stackCount / warpSize;

		// Set hash to the invalid value
		u32 warpHash = 0xffffffff;

		// Initial candidate position
		u32 warpHashCandidate = warpIndex % activeWarps;

		// Spin until the lock is acquired
		while( warpHash == 0xffffffff )
		{
			// The first active lane tries to lock the buffer for the whole warp
			if( laneIndex == firstLaneIndex )
			{
				// If it is successful, set the hash to the candidate value
				if( atomicCAS( &locks[warpHashCandidate], 0, 1 ) == 0 ) warpHash = warpHashCandidate;
			}
			// Try the next position
			warpHashCandidate = ( warpHashCandidate + 1 ) % activeWarps;
			// Exchange the hash withint the warp
			warpHash = __shfl( warpHash, firstLaneIndex );
		}

		// Save the acquired lock
		m_lock = &locks[warpHash];

		// Offset to the allocated buffer for individual threads in the warp 
		u32 offset = laneIndex + ( warpHash * warpSize ) * stackSize;

		// The stack buffer
		m_buffer = stackBuffer + offset;

		// The stack size
		m_size = stackSize;
	}

	// A destructor releasing the assigned memory
	__device__ ~Stack()
	{
		// Lane index
		u32 laneIndex = threadIdx.x & ( warpSize - 1 );
		// First active lane in the warp
		u32 firstLaneIndex = __ffsll( __ballot( true ) ) - 1;
		// Let the first lane release the lock
		if( laneIndex == firstLaneIndex ) atomicExch( m_lock, 0 );
	}

	// Pop method
	__device__ int pop() { return m_buffer[--m_index * warpSize]; }

	// Push method
	__device__ void push( int val ) { m_buffer[m_index++ * warpSize] = val; }

	// Empty method
	__device__ bool empty() const { return m_index == 0; }

  private:
	// Stack size
	u32 m_size;
	// Stack index
	int m_index = 0;
	// Pointer to the corresponding lock
	int* m_lock;
	// Pointer to the stack buffer
	int* m_buffer;
};

// A kernel counting, for a given query, how many leaves contain the same value
extern "C" __global__ void CountKernel( u32 size, u32 stackSize, u32 stackCount, int* stackBuffer, int* locks, const Node* nodes, const Leaf* leaves, int* queries, int* counts )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index >= size ) return;

	// Fetch query
	int query = queries[index];
	int count = 0;

	// Stack node indices
	Stack stack( stackSize, stackCount, stackBuffer, locks );

	// Push the root index
	stack.push( 0 );

	// Continue until stack is empty
	while( !stack.empty() )
	{
		int nodeIndex = stack.pop();
		// Internal node
		if( !isLeaf( nodeIndex ) )
		{
			// Fetch the internal node
			const Node& node = nodes[getNodeAddr( nodeIndex )];
			// Pivot splits the range covered by the node
			// If the query is greater or equal than the pivot then the value might be in the right subtree
			if( node.m_pivot <= query ) stack.push( node.m_rightIndex );
			// If the query is less or equal than the pivot then the value might be in the right subtree
			if( node.m_pivot >= query ) stack.push( node.m_leftIndex );
		}
		// Leaf
		else
		{
			// Fetch the leaf 
			const Leaf& leaf = leaves[getNodeAddr( nodeIndex )];
			// Increment the counter if the query is equal to the leaf value
			if( leaf.m_value == query ) ++count;
		}
	}
	// Write the count
	counts[index] = count;
}
