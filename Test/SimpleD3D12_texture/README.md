# simpleD3D12_texture - Simple D3D12 Oro Interop with Texture

## Description

Based this github project https://github.com/mprevot/CudaD3D12Update which demonstrates a DirectX 12 texture updated by cuda, through a cudaArray / surface2D.
This project has been converted to Orochi to make it compatible with both CUDA and HIP.
It's still a drafty project, but it should work correctly from HIP 6.0

Note: 2 manual modifications must be done in order to make this program work on Cuda:

- In the kernel file, hipSurfaceObject_t must be replaced by cudaSurfaceObject_t.

- The last argument of oroExternalMemoryGetMappedMipmappedArray seems to use a different structure on CUDA, it must me replaced by:
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC cuExtmemMipDesc{};
cuExtmemMipDesc.offset = 0;
cuExtmemMipDesc.arrayDesc.Width = texDesc.Width;
cuExtmemMipDesc.arrayDesc.Height = texDesc.Height;
cuExtmemMipDesc.arrayDesc.Depth = 0;
cuExtmemMipDesc.arrayDesc.Format = CU_AD_FORMAT_FLOAT;
cuExtmemMipDesc.arrayDesc.NumChannels = 4;
cuExtmemMipDesc.arrayDesc.Flags = 2;
cuExtmemMipDesc.numLevels = 1;

 