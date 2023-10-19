

extern "C" __global__ void texture_test( hipTextureObject_t tex_grid )
{
	if( threadIdx.x == 0 && blockIdx.x == 0 )
	{
		const auto feature_vector = tex2D<float4>( tex_grid, 0, 0 );
	}
}