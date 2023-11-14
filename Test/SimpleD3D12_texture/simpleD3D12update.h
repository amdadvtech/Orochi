#pragma once

#include "DX12CudaSample.h"
#include "ShaderStructs.h"
#include <chrono>
#include <sstream>
#include <string>
#include <algorithm>
#include <vector>


#include <Orochi/Orochi.h>
#include <Orochi/OrochiUtils.h>



typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::steady_clock::time_point TimePoint;

using namespace std;
using namespace DirectX;
using Microsoft::WRL::ComPtr;

#define CheckCudaErrors(val) Check((val), #val, __FUNCTION__, __FILE__, __LINE__)

typedef void(__stdcall* MessageChangedCallback)(const wchar_t* string);

class DX12CudaInterop : public DX12CudaSample
{
public:
	DX12CudaInterop(UINT width, UINT height, std::string name);
	virtual void OnInit();
	virtual void OnRender();
	virtual void OnDestroy();
	
	template <class T>
	auto Check(T result, const char* func, const char* caller, const char* file, int line) -> void;
	template <class ... T>
	auto LogMessage(T&& ... args) -> void;
	template <class T>
	auto WriteImageToFile(const char* filename, T* image) -> void;

private:
	static const UINT FrameCount = 2;
	size_t TextureHeight, TextureWidth;

	vector<wstring> Messages{};
	MessageChangedCallback LogMessageChangedCallback{};

	ComPtr<IDXGIFactory4> dxgiFactory;
	ComPtr<ID3D12Debug> debugController;

	// Pipeline objects.
	D3D12_VIEWPORT m_viewport;
	CD3DX12_RECT m_scissorRect;
	ComPtr<IDXGISwapChain3> m_swapChain;
	ComPtr<ID3D12Device> m_device;
	ComPtr<ID3D12Resource> m_renderTargets[FrameCount];
	ComPtr<ID3D12CommandAllocator> m_commandAllocators[FrameCount];
	ComPtr<ID3D12CommandQueue>   m_commandQueue;
	ComPtr<ID3D12RootSignature>  m_rootSignature;
	ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
	ComPtr<ID3D12DescriptorHeap> m_srvHeap;
	ComPtr<ID3D12PipelineState>  m_pipelineState;
	ComPtr<ID3D12GraphicsCommandList> m_commandList;
	UINT m_rtvDescriptorSize;
	
	// App resources.
	ComPtr<ID3D12Resource> m_vertexBuffer;
	ComPtr<ID3D12Resource> TextureArray;
	D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;

	// Synchronization objects.
	UINT m_frameIndex = 0;
	HANDLE m_fenceEvent = 0;
	ComPtr<ID3D12Fence> m_fence;
	UINT64 m_fenceValues[FrameCount];

	TimePoint lastTimePoint;

	// CUDA objects
	oroExternalMemoryHandleType m_externalMemoryHandleType = (oroExternalMemoryHandleType)0;
	oroExternalMemory m_externalMemory = 0;
	oroExternalSemaphore m_externalSemaphore = 0;
	oroStream m_streamToRun = 0;
	LUID m_dx12deviceluid{0,0};
	UINT m_cudaDeviceID = 0;
	UINT m_nodeMask = 0;
	float m_AnimTime = 0.0f;
	float timeStep{0.1f};
	oroSurfaceObject cuSurface{};
	UINT8 TextureChannels = 0;

	void LoadPipeline();
	void InitCuda();
	void LoadAssets();
	void UpdateCudaSurface();
	void PopulateCommandList();
	void MoveToNextFrame();
	void WaitForGpu();

	UINT m_oroDeviceID = 0;
	oroDevice  gOroDevice = 0;
	oroCtx gOroContext = nullptr;

};

template <class T>
auto DX12CudaInterop::Check(T result, const char* func, const char* caller, const char* file, int line) -> void
{
	if (result)
	{
		wstringstream o;
		auto f = string(file);
		replace(f.begin(), f.end(), '\\', '/');
		auto justfile = string(f.substr(f.find_last_of('/') + 1));

		const char* pStr = nullptr;
		oroGetErrorName((oroError)result,&pStr);
		o << caller << "(): " << func << " at " << justfile.c_str() << ":" << line << " [" << pStr << "]";
		
		LogMessage(L"%s\n", o.str().c_str());
	}
}

template<class ...T>
auto DX12CudaInterop::LogMessage(T&&... args) -> void
{
	//if (LogMessageChangedCallback != nullptr)
	{
		wchar_t updatedMessage[4096];
		swprintf_s(updatedMessage, forward<T>(args)...);
		Messages.push_back(*new wstring(updatedMessage));
		throw;
		//LogMessageChangedCallback(updatedMessage);
	}
}

//template <typename T>
//auto DX12CudaInterop::WriteImageToFile(const char* filename, T* image) -> void
//{
//	auto tif = TIFFOpen(filename, "w");
//	if (tif)
//	{
//		TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, TextureWidth);
//		TIFFSetField(tif, TIFFTAG_IMAGELENGTH, TextureHeight);
//		TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 3);
//		TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
//		TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
//		TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
//		TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
//		TIFFSetField(tif, TIFFTAG_SOFTWARE, "simpleD3D12-D3D12CudaSurf");
//
//		auto scanlineSize = TIFFScanlineSize(tif);
//		auto scanline = static_cast<T*>(_TIFFmalloc(scanlineSize));
//		auto stride = TextureWidth * 3;
//		for (auto y = 0; y < TextureHeight; y++)
//		{
//			memcpy(scanline, image + y * stride, scanlineSize);
//			TIFFWriteScanline(tif, scanline, y, 0);
//		}
//		TIFFFlushData(tif);
//		free(scanline);
//	}
//	TIFFClose(tif);
//}