#include <common/Common.h>
#include <LP.h>

extern "C" __global__ void test( LP_Concurrent<false> lp )
{ 
	if( ( threadIdx.x % 2 ) == 0)
	{
		lp.insert( threadIdx.x );
	}
}