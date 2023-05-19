#pragma once
#include <common/Common.h>

#if defined( __KERNELCC__ )
#define DEVICE __device__
#else
#define DEVICE
#endif

template<class T, bool isCpu>
struct Buffer
{
	T* m_data = nullptr;
	u32 m_size = 0;

#if !defined( __KERNELCC__ )
	void resize( u32 n ) 
	{ 
		if( isCpu )
		{
			if( m_data )
			{
				free( m_data );
			}
			m_data = (T*)malloc( n * sizeof( T ) );
		}
		else
		{
			if( m_data )
			{
				OrochiUtils::free( m_data );
			}
			OrochiUtils::malloc( m_data, n );
		}
		m_size = n;
	}
	~Buffer()
	{
		if( isCpu )
		{
			free( m_data );
			m_data = nullptr;
		}
		else
		{
			OrochiUtils::free( m_data );
			m_data = nullptr;
		}
	}
	void fillZero()
	{
		if( isCpu )
		{
			memset( m_data, 0, sizeof( T ) * m_size );
		}
		else
		{
			OrochiUtils::memset( m_data, 0, sizeof( T ) * m_size );
		}
	}

	void copyTo( Buffer<T, true>* buffer ) // to cpu
	{
		buffer->resize( size() );

		if( isCpu )
		{
			memcpy( buffer->m_data, m_data, sizeof( T ) * size() );
		}
		else
		{
			OrochiUtils::copyDtoH( buffer->m_data, m_data, size() );
		}
	}
	void copyTo( Buffer<T, false>* buffer ) // to gpu
	{
		buffer->resize( size() );

		if( isCpu )
		{
			OrochiUtils::copyHtoD( buffer->m_data, m_data, size() );
		}
		else
		{
			OrochiUtils::copyDtoD( buffer->m_data, m_data, size() );
		}
	}
#endif
	DEVICE
	u32 size() const { return m_size; }

	DEVICE 
	const T* data() const { return m_data; }

	DEVICE
	T* data() { return m_data; }

	DEVICE
	const T& operator[]( int index ) const { return m_data[index]; }

	DEVICE
	T& operator[]( int index ) { return m_data[index]; }
};

template<class T>
using BufferGPU = Buffer<T, false>;

template<class T>
using BufferCPU = Buffer<T, true>;


struct splitmix64
{
	u64 x = 0; /* The state can be seeded with any value. */

	DEVICE
	u64 next()
	{
		u64 z = ( x += 0x9e3779b97f4a7c15 );
		z = ( z ^ ( z >> 30 ) ) * 0xbf58476d1ce4e5b9;
		z = ( z ^ ( z >> 27 ) ) * 0x94d049bb133111eb;
		return z ^ ( z >> 31 );
	}
};

const int INT_PHI = 0x9e3779b9;
const int INV_INT_PHI = 0x144cbc89;

DEVICE
inline u32 hash( u32 x )
{
	x *= INT_PHI;
	x ^= x >> 16;
	return x;
}

DEVICE
inline u32 unhash( u32 x )
{
	x ^= x >> 16;
	x *= INV_INT_PHI;
	return x;
}

const u32 OCCUPIED_BIT = 1 << 31;
const u32 VALUE_MASK = ~OCCUPIED_BIT;

#if !defined( __KERNELCC__ )
inline u32 atomicCAS( u32* address, u32 compare, u32 val )
{
	return InterlockedCompareExchange( address, val, compare );
}
#endif

template <bool isCpu>
class LP_Concurrent
{
  public:
#if !defined( __KERNELCC__ )
	LP_Concurrent() {}
	LP_Concurrent( int n )
	{
		m_table.resize( n );
		m_table.fillZero();
	}
#endif
	DEVICE
	int home( u32 k ) const { return hash( k ) % m_table.size(); }

	enum InsertionResult
	{
		INSERTED,
		FOUND,
		OUT_OF_MEMORY
	};

	// k must be less than equal 0x7FFFFFFF
	DEVICE
	InsertionResult insert( u32 k )
	{
		u32 h = home( k );
		for( int i = 0; i < m_table.size(); i++ )
		{
			int location = ( h + i ) % m_table.size();
			u32 r = atomicCAS( &m_table[location], 0 /* empty */, k | OCCUPIED_BIT );
			if( r == 0 )
			{
				return INSERTED;
			}
			else if( r == ( k | OCCUPIED_BIT ) )
			{
				return FOUND;
			}
		}
		return OUT_OF_MEMORY;
	}

	DEVICE
	bool find( u32 k ) const
	{
		u32 h = home( k );
		for( int i = 0; i < m_table.size(); i++ )
		{
			int location = ( h + i ) % m_table.size();
			if( ( m_table[location] & OCCUPIED_BIT ) == 0 )
			{
				return false;
			}
			else if( m_table[location] == ( k | OCCUPIED_BIT ) )
			{
				return true;
			}
		}
		return false;
	}
#if !defined( __KERNELCC__ )
	std::set<u32> set() const
	{
		std::set<u32> s;
		for( int i = 0; i < m_table.size() ; i++)
		{
			auto value = m_table[i];
			if( value & OCCUPIED_BIT )
			{
				s.insert( value & VALUE_MASK );
			}
		}
		return s;
	}

	void print()
	{
		printf( "data=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			if( m_table[i] & OCCUPIED_BIT )
			{
				printf( "%03d, ", m_table[i] & VALUE_MASK );
			}
			else
			{
				printf( "---, " );
			}
		}
		printf( "\n" );

		printf( "home=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			if( m_table[i] & OCCUPIED_BIT )
			{
				printf( "%03d, ", home( m_table[i] & VALUE_MASK ) );
			}
			else
			{
				printf( "---, " );
			}
		}
		printf( "\n" );
	}

	// LP_Concurrent
	void copyTo( LP_Concurrent<false>* other )
	{
		m_table.copyTo( &other->m_table );
	}
	void copyTo( LP_Concurrent<true> *other )
	{
		m_table.copyTo( &other->m_table );
	}
#endif
	Buffer<u32, isCpu> m_table;
};

using LP_ConcurrentCPU = LP_Concurrent<true>;
using LP_ConcurrentGPU = LP_Concurrent<false>;