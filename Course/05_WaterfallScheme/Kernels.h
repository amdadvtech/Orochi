#include <common/Common.h>

// A kernel building a binary tree (a parallel version of TreeBuilder::build in Common.h) 
extern "C" __global__ void BuildTree( u32 size, int* input, int* taskQueue, int* counters, Node* nodes, Leaf* leaves )
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	
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

		// Task queue contains node indices to the input node buffer
		// Initially, the task queue contains only '1' in the first entry
		// Other entries are zero indicating that they are not valid tasks
		int task = taskQueue[index];

		// If the index is valid and the node has not been processed yet
		if( task != 0 && !done )
		{
			// Fetch the node
			Node node = nodes[index];
			int l = node.m_leftIndex;
			int r = node.m_rightIndex;
			int m = ( l + r ) / 2;
			node.m_pivot = input[m];

			// Count the number of internal nodes (0-2)
			u32 internalCount = 0;
			if( m - l > 1 ) ++internalCount;
			if( r - m > 1 ) ++internalCount;
			
			// Add the count to get the offset
			int childOffset = atomicAdd( &counters[0], internalCount );
			
			// Left child => internal node
			if( m - l > 1 )
			{
				// New child index
				int childIndex = childOffset++;
				// Update the node left child index
				node.m_leftIndex = childIndex;
				// Write child node
				nodes[childIndex] = { l, m, index };
				// Make sure that node has been flushed
				__threadfence();
				// Write child task
				taskQueue[childIndex] = 1; 
			}
			// Left child => leaf
			else
			{
				// Leaf index encoded as negative
				node.m_leftIndex = ~l;
				// Write the leaf
				leaves[l] = { input[l], index };
			}

			// Right child => internal node
			if( r - m > 1 )
			{
				// New child index
				int childIndex = childOffset;
				// Update the node right child index
				node.m_rightIndex = childIndex;
				// Write child node
				nodes[childIndex] = { m, r, index };
				// Make sure that node has been flushed
				__threadfence();
				// Write child task
				taskQueue[childIndex] = 1;
			}
			// Right child => leaf
			else
			{
				// Leaf index encoded as negative
				node.m_rightIndex = ~m;
				// Write the leaf
				leaves[m] = { input[m], index };
			}

			// Place the input node to the new position
			// Notice that the thread index correspond the node index
			nodes[index] = node;

			// Add the number of processed leaves to the global counter
			atomicAdd( &counters[1], 2 - internalCount );

			// Mark the thread as done
			done = true;
		}
		// We escape from the loop only when all threads in the warp are done
		if( __all( done ) ) break;
	}
}
