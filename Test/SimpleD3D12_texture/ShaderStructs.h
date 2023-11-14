#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <D3Dcompiler.h>
#include <DirectXMath.h>

using namespace DirectX;

struct TexVertex
{
	XMFLOAT3 position;
	XMFLOAT2 uv;
};
