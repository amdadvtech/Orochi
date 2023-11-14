#include "stdafx.h"
#include "simpleD3D12update.h"

_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
	DX12CudaInterop sample(2160, 1080, "D3D12-HIP Interop");
	return Win32Application::Run(&sample, hInstance, nCmdShow);
}
