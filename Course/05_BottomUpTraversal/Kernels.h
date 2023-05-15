#include <common/Common.h>

extern "C" __global__ void BottomUpTraversalKernel( u32 size, const Node* nodes, const Leaf* leaves, int* sums, int* counters )
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index >= size ) return;

	const Leaf& leaf = leaves[index];
	index = leaf.m_parent;

	while( index >= 0 && atomicAdd( &counters[index], 1 ) > 0 )
	{
		__threadfence();

		const Node& node = nodes[index];

		int sum = 0;
		if( node.m_left < 0 ) 
			sum += leaves[~node.m_left].m_value;
		else
			sum += sums[node.m_left];

		if( node.m_right < 0 ) 
			sum += leaves[~node.m_right].m_value;
		else
			sum += sums[node.m_right];

		sums[index] = sum;
		index = node.m_parent;
	}
}
