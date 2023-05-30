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
	void copyTo( T* ptr )
	{
		if( isCpu )
		{
			memcpy( ptr, m_data, sizeof( T ) * size() );
		}
		else
		{
			OrochiUtils::copyDtoH( ptr, m_data, size() );
		}
	}

	void** dataPtr() { return (void**)&m_data; }
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

#if !defined( __KERNELCC__ )
inline u32 atomicCAS( u32* address, u32 compare, u32 val )
{
	return InterlockedCompareExchange( address, val, compare );
}
#endif

template <bool isCpu>
class LP_Concurrent
{
	enum
	{
		OCCUPIED_BIT = 1 << 31,
		VALUE_MASK = ~OCCUPIED_BIT,
	};

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

	DEVICE
	float getOccupancy() const
	{
		int nOccupied = 0;
		for (int i = 0; i < m_table.size(); i++)
		{
			if( m_table[i] & OCCUPIED_BIT )
			{
				nOccupied++;
			}
		}
		return (float)nOccupied / m_table.size();
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

template<int isCpu>
class BLP_Concurrent
{
  public:
	enum
	{
		OCCUPIED_BIT = 1 << 31,
		LOCK_BIT = 1 << 30,
		VALUE_MASK = ~( OCCUPIED_BIT | LOCK_BIT ),
		EMPTY = 0,
		EMPTY_LOCKED = LOCK_BIT,
	};

#if !defined( __KERNELCC__ )
	BLP_Concurrent() {}
	BLP_Concurrent( int n )
	{
		m_table.resize( n );
		m_table.fillZero();
	}
#endif
	DEVICE
	u32 home( u32 k ) const
	{
		u32 hashK = hash( k );
		return static_cast<u32>( static_cast<u64>( hashK ) * m_table.size() / ( static_cast<u64>( 0xFFFFFFFF ) + 1 ) );
	}
	DEVICE
	bool find( u32 k, u32 hashK, u32 j ) const
	{
		int dir = hash( m_table[j] & VALUE_MASK ) < hashK ? +1 : -1;
		while( j < m_table.size() && ( m_table[j] & OCCUPIED_BIT ) && ( (i64)hash( m_table[j] & VALUE_MASK ) - (i64)hashK ) * dir < 0 )
		{
			j += dir;
		}
		return j < m_table.size() && ( m_table[j] & ( VALUE_MASK | OCCUPIED_BIT ) ) == ( k | OCCUPIED_BIT );
	}
	enum InsertionResult
	{
		INSERTED,
		FOUND,
		OUT_OF_MEMORY
	};

	// try to acquire a lock but it is ignored if the location is out of bounds
	// return true if it is succeeded.
	DEVICE
	bool tryLock( u32 location )
	{
		if( location < m_table.size() )
		{
			return atomicCAS( &m_table[location], EMPTY, EMPTY_LOCKED ) == EMPTY;
		}
		return true;
	}

	// unlock the acquired lock but it is ignored if the location is out of bounds
	DEVICE
	void unlock( u32 location )
	{
		if( location < m_table.size() )
		{
			atomicCAS( &m_table[location], EMPTY_LOCKED, EMPTY );
		}
	}

	// k must be less than equal 0x3FFFFFFF
	DEVICE
	InsertionResult insert( u32 k )
	{
	retry:
		u32 h = home( k );
		u32 oldval = atomicCAS( &m_table[h], EMPTY, k | OCCUPIED_BIT );
		if( oldval == EMPTY )
		{
			return INSERTED;
		}
		else if( oldval == ( k | OCCUPIED_BIT ) )
		{
			return FOUND;
		}
		else if( oldval == EMPTY_LOCKED )
		{
			goto retry;
		}

		u32 hashK = hash( k );
		if( find( k, hashK, h ) )
		{
			return FOUND;
		}

		u32 t_left = h - 1;
		u32 t_right = h + 1;

		while( t_left < m_table.size() && ( m_table[t_left] & OCCUPIED_BIT ) )
		{
			t_left--;
		}
		while( t_right < m_table.size() && ( m_table[t_right] & OCCUPIED_BIT ) )
		{
			t_right++;
		}

		if( m_table.size() <= t_left && m_table.size() <= t_right )
		{
			return OUT_OF_MEMORY;
		}

		// TryLock
		if( tryLock( t_left ) == false )
		{
			goto retry;
		}
		if( tryLock( t_right ) == false )
		{
			unlock( t_left );
			goto retry;
		}

		// It is possible to got some changes during lock process. So check again.
		if( find( k, hashK, h ) )
		{
			unlock( t_left );
			unlock( t_right );
			return FOUND;
		}

		int dir = hash( m_table[h] & VALUE_MASK ) < hashK ? +1 : -1;

		// if t_left is not empty, you can't move toward left
		if( m_table.size() <= t_left )
		{
			dir = -1;
		}
		// if t_right is not empty, you can't move toward right
		if( m_table.size() <= t_right )
		{
			dir = +1;
		}

		u32 j = 0 < dir ? t_left : t_right;
		while( j + dir < m_table.size() && ( m_table[j + dir] & OCCUPIED_BIT ) && ( (i64)hash( m_table[j + dir] & VALUE_MASK ) - (i64)hashK ) * dir < 0 )
		{
			m_table[j] = m_table[j + dir];
			j += dir;
		}

		m_table[j] = k | OCCUPIED_BIT;
		unlock( t_left );
		unlock( t_right );

		return INSERTED;
	}

	DEVICE
	bool find( u32 k ) const
	{
		u32 h = home( k );
		u32 hashK = hash( k );
		return find( k, hashK, h );
	}
#if !defined( __KERNELCC__ )
	std::set<u32> set() const
	{
		std::set<u32> s;
		for( int i = 0; i < m_table.size(); i++ )
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
				printf( "%02d, ", m_table[i] & ~OCCUPIED_BIT );
			}
			else
			{
				printf( "--, " );
			}
		}
		printf( "\nhome=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			printf( "%02d, ", home( m_table[i] & ~OCCUPIED_BIT ) );
		}
		printf( "\n" );

		printf( "hash=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			printf( "%x, ", hash( m_table[i] & VALUE_MASK ) );
		}
		printf( "\n" );
	}
#endif
	Buffer<u32, isCpu> m_table;
};

using BLP_ConcurrentCPU = BLP_Concurrent<true>;
using BLP_ConcurrentGPU = BLP_Concurrent<false>;
