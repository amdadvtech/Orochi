#include <common/Common.h>

extern "C" __global__ void BottomUpTraversalKernel( u32 size, const Node* nodes, const Leaf* leaves, int* sums, int* counters )
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index >= size ) return;

	const Leaf& leaf = leaves[index];
	index = leaf.getParentIndex();

	while( index >= 0 && atomicAdd( &counters[index], 1 ) > 0 )
	{
		__threadfence();

		const Node& node = nodes[index];

		int sum = 0;
		if( node.isLeftLeaf() ) 
			sum += leaves[node.getLeftIndex()].getValue();
		else
			sum += sums[node.getLeftIndex()];

		if( node.isRightLeaf() ) 
			sum += leaves[node.getRightIndex()].getValue();
		else
			sum += sums[node.getRightIndex()];

		sums[index] = sum;
		index = node.getParentIndex();
	}
}
