#include <common/Common.h>

// Bottom-up traversal kernel using atomic counters in internal nodes summing up values in leaves
extern "C" __global__ void BottomUpTraversalKernel( u32 size, const Node* nodes, const Leaf* leaves, int* sums, int* counters )
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index >= size ) return;

	// Fetch the corresponding leaf
	const Leaf& leaf = leaves[index];
	// Go to its parent
	index = leaf.m_parentAddr;

	// Negative index means that we reached the root node that does not have parent
	// The second condition ensures that only the second thread continues
	// To be sure that all nodes in the subtree have been processed
	while( index >= 0 && atomicAdd( &counters[index], 1 ) > 0 )
	{
		// We are modifying 'sums' which is in global memory
		// Thus, we use thread fence to make sure that cashes have been flushed
		__threadfence();

		// Fetch the node
		const Node& node = nodes[index];

		// Sum for this node
		int sum = 0;

		// Process left chid
		if( node.isLeftLeaf() ) 
			// From leaf fetch the value
			sum += leaves[node.getLeftAddr()].m_value;
		else
			// From internal load the corresponding sum
			sum += sums[node.getLeftAddr()];

		// Process right chid
		if( node.isRightLeaf() ) 
			// From leaf fetch the value
			sum += leaves[node.getRightAddr()].m_value;
		else
			// From internal load the corresponding sum
			sum += sums[node.getRightAddr()];

		// Write the sum for this node
		sums[index] = sum;

		// Go to parent
		index = node.m_parentAddr;

		// Make sure that the operations are done in order
		__threadfence();
	}
}
