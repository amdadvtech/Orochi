#include <common/Common.h>

// A kernel reodering a binary tree to the breadth-first-search (BFS) layout
extern "C" __global__ void ConvertToBFSKernel( u32 size, const Node* inNodes, const Leaf* leaves, int* taskQueue, int* counters, Node* outNodes )
{
	u32 index = threadIdx.x + blockDim.x * blockIdx.x;
	
	// A flag indicating whether the thread is done
	// This is needed to postpone exit until all threads in the warp are done
	// For architectures wihtout 'independent thread scheduling'
	bool done = false;

	// We are done if the all leaves haven been reached
	while( atomicAdd( &counters[1], 0 ) < size )
	{
		// We are modifying a task queue which resides in the global memory
		// Thus, we use thread fence to make sure that cashes have been flushed
		__threadfence();

		// Skip threads that are out-of-bound
		if( index >= size - 1 ) continue;

		// Taks queue contains node indices to the input node buffer
		// Initially, the task queue contains only '0' in the first entry
		// Aother entries are negative indicating that they are not valid tasks
		int nodeIndex = taskQueue[index];

		// If the index is valid and the node has not been processedyet
		if( nodeIndex >= 0 && !done )
		{
			// Fetch the input node
			Node node = inNodes[nodeIndex];

			// Count the number of internal nodes (0-2)
			u32 internalCount = 0;
			if( !node.isLeftLeaf() ) ++internalCount;
			if( !node.isRightLeaf() ) ++internalCount;
			
			// Add the count to get the offset
			int childOffset = atomicAdd( &counters[0], internalCount );
			
			// Left child => internal node
			if( !node.isLeftLeaf() )
			{
				// New child index
				int childIndex = childOffset++;
				// We use the same index for the task
				taskQueue[childIndex] = node.m_leftIndex; 
				// Update the node left child index
				node.setLeftIndex( childIndex );
			}

			// Right child => internal node
			if( !node.isRightLeaf() )
			{
				// New child index
				int childIndex = childOffset;
				// We use the same index for the task
				taskQueue[childIndex] = node.m_rightIndex;
				// Update the node right child index
				node.setRightIndex( childIndex );
			}

			// Place the input node to the new position
			// Notice that the thread index correspond the output index
			outNodes[index] = node;

			// Add the number of processed leaves to the global counter
			atomicAdd( &counters[1], 2 - internalCount );

			// Mark the thread as done
			done = true;
		}
		// We escape from the loop only when all threads in the warp are done
		if( __all( done ) ) break;
	}
}
