#include <common/Common.h>

extern "C" __global__ void ConvertToBFSKernel( u32 size, const Node* inNodes, const Leaf* leaves, int* taskQueue, int* counters, Node* outNodes )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	
	bool done = false;
	while( atomicAdd( &counters[1], 0 ) < size )
	{
		__threadfence();

		if( index >= size - 1 ) continue;

		int nodeIndex = taskQueue[index];
		if( nodeIndex >= 0 && !done )
		{
			Node node = inNodes[nodeIndex];

			u32 internalCount = 0;
			if( node.m_left >= 0 ) ++internalCount;
			if( node.m_right >= 0 ) ++internalCount;
			
			int childOffset = atomicAdd( &counters[0], internalCount );
			
			if( node.m_left >= 0 )
			{
				int childIndex = childOffset++;
				taskQueue[childIndex] = node.m_left; 
				node.m_left = childIndex;
			}

			if( node.m_right >= 0 )
			{
				int childIndex = childOffset;
				taskQueue[childIndex] = node.m_right;
				node.m_right = childIndex;
			}

			outNodes[index] = node;

			atomicAdd( &counters[1], 2 - internalCount );
			done = true;
		}
		if( !__any( !done ) ) break;
	}
}
