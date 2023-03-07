__kernel void testKernel( int x ) 
{ 
	int idx = get_global_id( 0 );
	printf( "%d: %d\n", idx, x );
}

