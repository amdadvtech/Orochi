#pragma once

#include <Orochi/Orochi.h>
#include <string>
#include <unordered_map>

namespace Oro
{

class RadixSort
{
  public:
	typedef unsigned int u32;

	enum Flag
	{
		FLAG_LOG = 1 << 0,
	};

	RadixSort();

	// Allow move but disallow copy.
	RadixSort( RadixSort&& ) = default;
	RadixSort& operator=( RadixSort&& ) = default;
	RadixSort( const RadixSort& ) = delete;
	RadixSort& operator=( const RadixSort& ) = delete;
	~RadixSort();

	void configure( oroDevice device, u32& tempBufferSizeOut );

	void setFlag( Flag flag );

	void sort( u32* src, u32* dst, int n, int startBit, int endBit, u32* tempBuffer );

  private:
	void sort1pass( u32* src, u32* dst, int n, int startBit, int endBit, int* tmps );

	void compileKernels( oroDevice device );

  private:
	int m_nWGsToExecute{ 4 };
	Flag m_flags;

	enum class Kernel
	{
		COUNT,
		COUNT_REF,
		SCAN_SINGLE_WG,
		SCAN_PARALLEL,
		SORT,
		SORT_REF,
		SORT_SINGLE_PASS
	};

	std::unordered_map<Kernel, oroFunction> oroFunctions;

	/// @brief  The enum class which indicates the selected algorithm of prefix scan.
	enum class ScanAlgo
	{
		SCAN_CPU,
		SCAN_GPU_SINGLE_WG,
		SCAN_GPU_PARALLEL,
	};

	constexpr static auto selectedScanAlgo{ ScanAlgo::SCAN_GPU_PARALLEL };

	int* m_partialSum{ nullptr };
	bool* m_isReady{ nullptr };
};

}; // namespace Oro
