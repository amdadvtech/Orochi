#include <common/Common.h>

extern "C" __global__ void BottomUpTraversalKernel( u32 size, const Node* nodes, const Leaf* leaves, int* sums, int* counters )
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index >= size ) return;

	const Leaf& leaf = leaves[index];
	index = leaf.m_parentAddr;

	while( index >= 0 && atomicAdd( &counters[index], 1 ) > 0 )
	{
		__threadfence();

		const Node& node = nodes[index];

		int sum = 0;
		if( node.isLeftLeaf() ) 
			sum += leaves[node.getLeftAddr()].m_value;
		else
			sum += sums[node.getLeftAddr()];

		if( node.isRightLeaf() ) 
			sum += leaves[node.getRightAddr()].m_value;
		else
			sum += sums[node.getRightAddr()];

		sums[index] = sum;
		index = node.m_parentAddr;
	}
}
