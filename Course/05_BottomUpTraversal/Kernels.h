#include <common/Common.h>

extern "C" __global__ void BottomUpTraversalKernel( u32 size, Node* nodes, Leaf* leaves )
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index < size ) return;

	Leaf& leaf = leaves[index];
	index = leaf.m_parent;

	while( atomicAdd( &nodes[index].m_counter, 1 ) > 0 )
	{
		__threadfence();

		Node& node = nodes[index];

		if( node.m_left < 0 )
			node.m_sum += leaves[~node.m_left].m_value;
		else
			node.m_sum += nodes[node.m_left].m_sum;

		if( node.m_right < 0 )
			node.m_sum += leaves[~node.m_right].m_value;
		else
			node.m_sum += nodes[node.m_right].m_sum;

		index = node.m_parent;
		if( index < 0 ) break;
	}
}
