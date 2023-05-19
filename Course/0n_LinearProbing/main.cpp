#include <common/Common.h>
#include <iostream>
#include <memory>
#include <set>
#include <random>
#include <mutex>
#include <Windows.h>
#include <ppl.h>

// Reference
// [0] Ordered hash tables ( original idea of BLP, very old )
// [1] A Concurrent Bidirectional Linear Probing Algorithm
// [2] A Parallel Compact Hash Table ( after [1] )

// very nice code example:
// https://github.com/senderista/hashtable-benchmarks

struct splitmix64
{
	uint64_t x = 0; /* The state can be seeded with any value. */
	uint64_t next()
	{
		uint64_t z = ( x += 0x9e3779b97f4a7c15 );
		z = ( z ^ ( z >> 30 ) ) * 0xbf58476d1ce4e5b9;
		z = ( z ^ ( z >> 27 ) ) * 0x94d049bb133111eb;
		return z ^ ( z >> 31 );
	}
};

const int INT_PHI = 0x9e3779b9;
const int INV_INT_PHI = 0x144cbc89;
uint32_t hash( uint32_t x )
{
	x *= INT_PHI;
	x ^= x >> 16;
	return x;
}
uint32_t unhash( uint32_t x )
{
	x ^= x >> 16;
	x *= INV_INT_PHI;
	return x;
}

const uint32_t OCCUPIED_BIT = 1 << 31;
const uint32_t VALUE_MASK = ~OCCUPIED_BIT;
using u32 = uint32_t;

inline u32 atomicCAS( u32* address, u32 compare, u32 val )
{
	return InterlockedCompareExchange( address, val, compare );
}
//inline u32 atomicRead( u32* address ) 
//{
//	return InterlockedExchangeAdd( address, 0 );
//}

class LP
{
  public:
	LP( int n ) : m_table( n ) {}

	int home( uint32_t k ) const { return hash( k ) % m_table.size(); }

	// k must be less than equal 0x7FFFFFFF
	int insert( uint32_t k )
	{
		uint32_t h = home( k );
		for( int i = 0; i < m_table.size(); i++ )
		{
			int location = ( h + i ) % m_table.size();

			if( ( m_table[location] & OCCUPIED_BIT ) == 0 )
			{
				m_table[location] = k | OCCUPIED_BIT;
				return location;
			}
			else if( ( m_table[location] & VALUE_MASK ) == k )
			{
				return location;
			}
		}
		return -1;
	}
	int find( uint32_t k ) const
	{
		uint32_t h = home( k );
		for( int i = 0; i < m_table.size(); i++ )
		{
			int location = ( h + i ) % m_table.size();
			if( ( m_table[location] & OCCUPIED_BIT ) == 0 )
			{
				return -1;
			}
			else if( m_table[location] == ( k | OCCUPIED_BIT ) )
			{
				return location;
			}
		}
		return -1;
	}
	std::set<uint32_t> set() const
	{
		std::set<uint32_t> s;
		for( auto value : m_table )
		{
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
	std::vector<uint32_t> m_table;
};

// find on concurrent and serial have different behavior, will fix later.
class StdSet_Concurrent
{
  public:
	StdSet_Concurrent( int n = 0 ){
	}
	void insert( u32 k )
	{
		std::lock_guard lock( m_mu );
		m_set.insert( k );
	}

	bool find( u32 k ) const
	{
		std::lock_guard lock( m_mu );
		return m_set.count( k ) != 0;
	}
	std::set<u32> set() const
	{
		std::lock_guard lock( m_mu );
		return m_set;
	}
	std::set<u32> m_set;
	mutable std::mutex m_mu;

	// int m_table[1];
};

class LP_Concurrent
{
  public:
	LP_Concurrent( int n ) : m_table( n ) {}

	int home( u32 k ) const { return hash( k ) % m_table.size(); }

	enum InsertionResult
	{
		INSERTED,
		FOUND,
		OUT_OF_MEMORY
	};
	// k must be less than equal 0x7FFFFFFF
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
			else if (r == (k | OCCUPIED_BIT))
			{
				return FOUND;
			}
		}
		return OUT_OF_MEMORY;
	}

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
	std::set<u32> set() const
	{
		std::set<u32> s;
		for( auto value : m_table )
		{
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
	std::vector<u32> m_table;
};

class RH
{
  public:
	RH( int n ) : m_table( n ) {}

	uint32_t home( uint32_t h ) const { return (uint64_t)h * m_table.size() / ( (uint64_t)UINT_MAX + 1 ); }

	// probe sequence lengths
	uint32_t psl( int i, uint32_t h )
	{
		/*
		PSL =
		 2  3  4     1
		[ ][ ][ ][h][ ]
		*/
		int homeLocation = (int)home( h );

		if( i < homeLocation )
		{
			// e.g. h 3 - 5 = -2
			homeLocation = homeLocation - (int)m_table.size();
		}

		return i - homeLocation;
	}

	// k must be less than equal 0x7FFFFFFF
	int insert( uint32_t k )
	{
		uint32_t h = home( hash( k ) );
		int PSL = 0;
		for( int i = 0; i < m_table.size(); i++ )
		{
			int location = ( h + i ) % m_table.size();

			if( ( m_table[location] & OCCUPIED_BIT ) == 0 )
			{
				m_table[location] = k | OCCUPIED_BIT;
				return location;
			}
			else if( ( m_table[location] & VALUE_MASK ) == k )
			{
				return location;
			}
			int thePSL = psl( location, hash( m_table[location] & VALUE_MASK ) );
			if( thePSL < PSL )
			{
				uint32_t theValue = m_table[location] & VALUE_MASK;
				m_table[location] = k | OCCUPIED_BIT;

				PSL = thePSL;
				k = theValue;
			}
			PSL++;
		}
		return -1;
	}
	int find( uint32_t k ) const
	{
		uint32_t h = home( hash( k ) );
		for( int i = 0; i < m_table.size(); i++ )
		{
			int location = ( h + i ) % m_table.size();
			if( ( m_table[location] & OCCUPIED_BIT ) == 0 )
			{
				return -1;
			}
			else if( m_table[location] == ( k | OCCUPIED_BIT ) )
			{
				return location;
			}
		}
		return -1;
	}
	std::set<uint32_t> set() const
	{
		std::set<uint32_t> s;
		for( auto value : m_table )
		{
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

		printf( "PSL =" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			if( m_table[i] & OCCUPIED_BIT )
			{
				printf( "%03d, ", psl( i, hash( m_table[i] & VALUE_MASK ) ) );
			}
			else
			{
				printf( "---, " );
			}
		}
		printf( "\n" );
	}
	std::vector<uint32_t> m_table;
};
class BLP
{
  public:
	BLP( int n ) : m_table( n ) {}

	uint32_t home( uint32_t k ) const
	{
		uint32_t v = hash( k );
		return (uint64_t)v * m_table.size() / ( (uint64_t)UINT_MAX + 1 );
	}

	// k must be less than equal 0x7FFFFFFF
	int insert( uint32_t k )
	{
		uint32_t h = home( k );
		if( ( m_table[h] & OCCUPIED_BIT ) == 0 )
		{
			m_table[h] = k | OCCUPIED_BIT;
			return h;
		}

		if( m_table[h] == ( k | OCCUPIED_BIT ) )
		{
			return h;
		}

		uint32_t hashK = hash( k );

		bool moveTowardLeft = hash( m_table[h] & VALUE_MASK ) < hashK;

		for( int iter = 0; iter < 2; iter++ )
		{
			int j = h;
			if( moveTowardLeft )
			{
				// find
				if( iter == 0 )
				{
					while( j + 1 < m_table.size() && m_table[j + 1] & OCCUPIED_BIT && hash( m_table[j + 1] & VALUE_MASK ) <= hashK )
					{
						j++;
					}
					if( ( m_table[j] & VALUE_MASK ) == k )
					{
						return j;
					}
					j = h;
				}

				// example: hash(k) = 8
				// 7 < hash(k)
				// [ ][ ][3][6][7][ ]
				//     |<-------j
				// The elements are too right-shifed.
				// find empty location on the left
				while( 0 < j && m_table[j] & OCCUPIED_BIT )
				{
					j--;
				}

				if( m_table[j] & OCCUPIED_BIT )
				{
					// No empty space in this dir. Try other direction
					moveTowardLeft = !moveTowardLeft;
					continue;
				}

				// move toword left while T[j+1] < hashK
				//     +--+
				//     v  |
				// [ ][3][3][6][7][ ]
				//     j
				while( j + 1 < m_table.size() && ( m_table[j + 1] & OCCUPIED_BIT ) && hash( m_table[j + 1] & VALUE_MASK ) < hashK )
				{
					m_table[j] = m_table[j + 1];
					j++;
				}

				// [ ][3][6][7][7][ ]
				//              j
			}
			else // hashK <= hash( m_table[h] & VALUE_MASK );
			{
				// find
				if( iter == 0 )
				{
					while( 0 <= j - 1 && m_table[j - 1] & OCCUPIED_BIT && hashK <= hash( m_table[j - 1] & VALUE_MASK ) )
					{
						j--;
					}
					if( ( m_table[j] & VALUE_MASK ) == k )
					{
						return j;
					}
					j = h;
				}

				// example: hash(k) = 5
				// hash(k) < 6
				// [ ][ ][ ][2][6][7][8][ ]
				//              h
				// The elements are too left-shifed.
				// find empty location on the right
				while( j + 1 < m_table.size() && m_table[j] & OCCUPIED_BIT )
				{
					j++;
				}

				if( m_table[j] & OCCUPIED_BIT )
				{
					// No empty space in this dir. Try other direction
					moveTowardLeft = !moveTowardLeft;
					continue;
				}

				// move toword right while hashK < T[j-1]
				//                    +--+
				//                    |  v
				// [ ][ ][ ][2][6][7][8][8]
				//                       j
				while( 0 <= j - 1 && ( m_table[j - 1] & OCCUPIED_BIT ) && hashK < hash( m_table[j - 1] & VALUE_MASK ) )
				{
					m_table[j] = m_table[j - 1];
					j--;
				}

				// [ ][ ][ ][2][6][6][7][8]
				//              j
			}

			m_table[j] = k | OCCUPIED_BIT;
			return j;
		}
		return -1;
	}
	int find( uint32_t k ) const
	{
		uint32_t h = home( k );
		if( ( m_table[h] & OCCUPIED_BIT ) == 0 )
		{
			return -1;
		}
		if( m_table[h] == ( k | OCCUPIED_BIT ) )
		{
			return h;
		}

		uint32_t hashK = hash( k );
		bool moveTowardLeft = hash( m_table[h] & VALUE_MASK ) < hashK;
		int j = h;
		if( moveTowardLeft )
		{
			while( j + 1 < m_table.size() && m_table[j + 1] & OCCUPIED_BIT && hash( m_table[j + 1] & VALUE_MASK ) <= hashK )
			{
				j++;
			}
			if( m_table[j] == ( k | OCCUPIED_BIT ) )
			{
				return j;
			}
		}
		else
		{
			while( 0 <= j - 1 && m_table[j - 1] & OCCUPIED_BIT && hashK <= hash( m_table[j - 1] & VALUE_MASK ) )
			{
				j--;
			}
			if( m_table[j] == ( k | OCCUPIED_BIT ) )
			{
				return j;
			}
		}
		return -1;
	}

	std::set<uint32_t> set() const
	{
		std::set<uint32_t> s;
		for( auto value : m_table )
		{
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
	std::vector<uint32_t> m_table;
};

class BLPZeroEmpty
{
  public:
	BLPZeroEmpty( int n ) : m_table( n ) {}

	uint32_t home( uint32_t hashValue ) const { return (uint64_t)hashValue * m_table.size() / ( (uint64_t)UINT_MAX + 1 ); }

	// k must be less than equal 0x7FFFFFFF
	int insert( uint32_t k )
	{
		k++;

		uint32_t hashK = hash( k );
		uint32_t h = home( hashK );
		if( m_table[h] == 0 )
		{
			m_table[h] = hashK;
			return h;
		}
		else if( m_table[h] == hashK )
		{
			return h;
		}

		bool moveTowardLeft = m_table[h] < hashK;

		for( int iter = 0; iter < 2; iter++ )
		{
			int j = h;
			if( moveTowardLeft )
			{
				// find
				if( iter == 0 )
				{
					while( j + 1 < m_table.size() && m_table[j + 1] && m_table[j + 1] <= hashK )
					{
						j++;
					}
					if( m_table[j] == hashK )
					{
						return j;
					}
					j = h;
				}

				// example: hash(k) = 8
				// 7 < hash(k)
				// [ ][ ][3][6][7][ ]
				//     |<-------j
				// The elements are too right-shifed.
				// find empty location on the left
				while( 0 < j && m_table[j] )
				{
					j--;
				}

				if( m_table[j] )
				{
					// No empty space in this dir. Try other direction
					moveTowardLeft = !moveTowardLeft;
					continue;
				}

				// move toword left while T[j+1] < hashK
				//     +--+
				//     v  |
				// [ ][3][3][6][7][ ]
				//     j
				while( j + 1 < m_table.size() && m_table[j + 1] && m_table[j + 1] < hashK )
				{
					m_table[j] = m_table[j + 1];
					j++;
				}

				// [ ][3][6][7][7][ ]
				//              j
			}
			else // hashK <= m_table[h];
			{
				// find
				if( iter == 0 )
				{
					while( 0 <= j - 1 && m_table[j - 1] && hashK <= m_table[j - 1] )
					{
						j--;
					}
					if( m_table[j] == hashK )
					{
						return j;
					}
					j = h;
				}

				// example: hash(k) = 5
				// hash(k) < 6
				// [ ][ ][ ][2][6][7][8][ ]
				//              h
				// The elements are too left-shifed.
				// find empty location on the right
				while( j + 1 < m_table.size() && m_table[j] )
				{
					j++;
				}

				if( m_table[j] )
				{
					// No empty space in this dir. Try other direction
					moveTowardLeft = !moveTowardLeft;
					continue;
				}

				// move toword right while hashK < T[j-1]
				//                    +--+
				//                    |  v
				// [ ][ ][ ][2][6][7][8][8]
				//                       j
				while( 0 <= j - 1 && m_table[j - 1] && hashK < m_table[j - 1] )
				{
					m_table[j] = m_table[j - 1];
					j--;
				}

				// [ ][ ][ ][2][6][6][7][8]
				//              j
			}

			m_table[j] = hashK;
			return j;
		}
		return -1;
	}
	int find( uint32_t k ) const
	{
		k++;

		uint32_t hashK = hash( k );
		uint32_t h = home( hashK );
		if( m_table[h] == 0 )
		{
			return -1;
		}
		else if( m_table[h] == hashK )
		{
			return h;
		}

		bool moveTowardLeft = m_table[h] < hashK;
		int j = h;
		if( moveTowardLeft )
		{
			while( j + 1 < m_table.size() && m_table[j + 1] && m_table[j + 1] <= hashK )
			{
				j++;
			}
			if( m_table[j] == hashK )
			{
				return j;
			}
		}
		else
		{
			while( 0 <= j - 1 && m_table[j - 1] && hashK <= m_table[j - 1] )
			{
				j--;
			}
			if( m_table[j] == hashK )
			{
				return j;
			}
		}
		return -1;
	}

	std::set<uint32_t> set() const
	{
		std::set<uint32_t> s;
		for( auto value : m_table )
		{
			if( value )
			{
				s.insert( unhash( value ) - 1 /* back to the original */ );
			}
		}
		return s;
	}

	void print()
	{
		printf( "data=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			if( m_table[i] )
			{
				printf( "%02d, ", unhash( m_table[i] ) - 1 /* back to the original */ );
			}
			else
			{
				printf( "--, " );
			}
		}
		printf( "\nhome=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			printf( "%02d, ", home( m_table[i] ) );
		}
		printf( "\n" );

		printf( "hash=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			printf( "%x, ", m_table[i] );
		}
		printf( "\n" );
	}
	std::vector<uint32_t> m_table;
};

class BLPZeroEmptyBranchless
{
  public:
	BLPZeroEmptyBranchless( int n ) : m_table( n ) {}

	uint32_t home( uint32_t hashValue ) const { return (uint64_t)hashValue * m_table.size() / ( (uint64_t)UINT_MAX + 1 ); }

	// k must be less than equal 0x7FFFFFFF
	int insert( uint32_t k )
	{
		k++;

		uint32_t hashK = hash( k );
		uint32_t h = home( hashK );
		if( m_table[h] == 0 )
		{
			m_table[h] = hashK;
			return h;
		}
		else if( m_table[h] == hashK )
		{
			return h;
		}

		bool moveTowardLeft = m_table[h] < hashK;

		for( int iter = 0; iter < 2; iter++ )
		{
			int j = h;
			if( moveTowardLeft )
			{
				// find
				if( iter == 0 )
				{
					while( j + 1 < m_table.size() && m_table[j + 1] && m_table[j + 1] <= hashK )
					{
						j++;
					}
					if( m_table[j] == hashK )
					{
						return j;
					}
					j = h;
				}

				// example: hash(k) = 8
				// 7 < hash(k)
				// [ ][ ][3][6][7][ ]
				//     |<-------j
				// The elements are too right-shifed.
				// find empty location on the left
				while( 0 < j && m_table[j] )
				{
					j--;
				}

				if( m_table[j] )
				{
					// No empty space in this dir. Try other direction
					moveTowardLeft = !moveTowardLeft;
					continue;
				}

				// move toword left while T[j+1] < hashK
				//     +--+
				//     v  |
				// [ ][3][3][6][7][ ]
				//     j
				while( j + 1 < m_table.size() && m_table[j + 1] && m_table[j + 1] < hashK )
				{
					m_table[j] = m_table[j + 1];
					j++;
				}

				// [ ][3][6][7][7][ ]
				//              j
			}
			else // hashK <= m_table[h];
			{
				// find
				if( iter == 0 )
				{
					while( 0 <= j - 1 && m_table[j - 1] && hashK <= m_table[j - 1] )
					{
						j--;
					}
					if( m_table[j] == hashK )
					{
						return j;
					}
					j = h;
				}

				// example: hash(k) = 5
				// hash(k) < 6
				// [ ][ ][ ][2][6][7][8][ ]
				//              h
				// The elements are too left-shifed.
				// find empty location on the right
				while( j + 1 < m_table.size() && m_table[j] )
				{
					j++;
				}

				if( m_table[j] )
				{
					// No empty space in this dir. Try other direction
					moveTowardLeft = !moveTowardLeft;
					continue;
				}

				// move toword right while hashK < T[j-1]
				//                    +--+
				//                    |  v
				// [ ][ ][ ][2][6][7][8][8]
				//                       j
				while( 0 <= j - 1 && m_table[j - 1] && hashK < m_table[j - 1] )
				{
					m_table[j] = m_table[j - 1];
					j--;
				}

				// [ ][ ][ ][2][6][6][7][8]
				//              j
			}

			m_table[j] = hashK;
			return j;
		}
		return -1;
	}
	int find( uint32_t k ) const 
	{
		k++;

		uint32_t hashK = hash( k );
		uint32_t h = home( hashK );

		if( m_table[h] == 0 )
		{
			return -1;
		}
		else if( m_table[h] == hashK )
		{
			return h;
		}
		int d = m_table[h] < hashK ? +1 : -1;
		int j = h;
		while( 0 <= j + d && j + d < m_table.size() && m_table[j + d] && ( (int64_t)m_table[j + d] - (int64_t)hashK ) * d <= 0 )
		{
			j += d;
		}
		if( m_table[j] == hashK )
		{
			return j;
		}
		return -1;
	}

	std::set<uint32_t> set() const
	{
		std::set<uint32_t> s;
		for( auto value : m_table )
		{
			if( value )
			{
				s.insert( unhash( value ) - 1 /* back to the original */ );
			}
		}
		return s;
	}

	void print()
	{
		printf( "data=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			if( m_table[i] )
			{
				printf( "%02d, ", unhash( m_table[i] ) - 1 /* back to the original */ );
			}
			else
			{
				printf( "--, " );
			}
		}
		printf( "\nhome=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			printf( "%02d, ", home( m_table[i] ) );
		}
		printf( "\n" );

		printf( "hash=" );
		for( int i = 0; i < m_table.size(); i++ )
		{
			printf( "%x, ", m_table[i] );
		}
		printf( "\n" );
	}
	std::vector<uint32_t> m_table;
};

template <class T>
void runTest( )
{
	uint32_t k = unhash( 0 );

	int NBuckets = 1000;
	int Numbers = 10000;
	splitmix64 rnd;

	for( int i = 0; i < 10000; i++ )
	{
		T lp( NBuckets );
		std::set<uint32_t> s;
		for( int j = 0; j < NBuckets * 0.75; j++ )
		{
			uint32_t v = rnd.next() % Numbers;
			s.insert( v );
			lp.insert( v );
		}
		OROASSERT( s == lp.set(), 0 );

		for( int i = 0; i < NBuckets * 0.75; i++ )
		{
			uint32_t v = rnd.next() % Numbers;
			bool found0 = s.count( v ) != 0;
			bool found1 = lp.find( v ) != -1;
			OROASSERT( found0 == found1, 0 );
		}
	}
}



template<class T>
void runPerfTest( )
{
	int NBuckets = 1000;
	int Numbers = 10000;
	double loadFactor = 0.75;

	Stopwatch sw;
	sw.start();

	int nfound = 0;
	splitmix64 rnd;
	for( int i = 0; i < 100000; i++ )
	{
		T lp( NBuckets );
		for( int j = 0; j < NBuckets * loadFactor; j++ )
		{
			uint32_t v = rnd.next() % Numbers;
			lp.insert( v );
		}

		for( int i = 0; i < NBuckets * loadFactor; i++ )
		{
			uint32_t v = rnd.next() % Numbers;
			bool found = lp.find( v ) != -1;
			if( found )
			{
				nfound++;
			}
		}
	}

	sw.stop();
	printf( "%s %f ms, %d\n", typeid( T ).name(), sw.getMs(), nfound );
}

template <class T>
void runConcurrentTest( )
{ 
	int NThreads = 32;
	int NBuckets  = 10000;
	int Numbers = 1000000;
	double loadFactor = 0.75;

	for (int k = 0; k < 100; k++ )
	{
		StdSet_Concurrent truth;
		T storage( NBuckets );

		for( int i = 0; i < NThreads; i++ )
		{
			int nItemPerThread = NBuckets * loadFactor / NThreads;
			concurrency::parallel_for( 0, NThreads, [k, nItemPerThread, Numbers, &truth, &storage](int index) 
			{
				splitmix64 rnd;
				rnd.x = k * 1000000 + index;
				for( int j = 0; j < nItemPerThread; j++ )
				{
					uint32_t v = rnd.next() % Numbers;
					storage.insert( v );
					truth.insert( v );
				}
			} );
		}
		OROASSERT( truth.set() == storage.set(), 0 );

		splitmix64 rnd;
		for( int i = 0; i < NBuckets; i++ )
		{
			uint32_t v = rnd.next() % Numbers;
			OROASSERT( storage.find( v ) == truth.find( v ), 0 );
		}
	}
}
template <class T>
void runConcurrentPerfTest()
{ 
	int NThreads = 32;
	int NBuckets = 100000;
	int Numbers = 10000000;
	double loadFactor = 0.75;


	int nfound = 0;

	float timeInsertion = 0;
	float timeFind = 0;

	Stopwatch sw;
	sw.start();
	
	int a = 0;

	for (int k = 0; k < 512; k++)
	{
		T storage( NBuckets );
		for( int i = 0; i < NThreads; i++ )
		{
			int nItemPerThread = NBuckets * loadFactor / NThreads;
			concurrency::parallel_for( 0, NThreads, [k, nItemPerThread, Numbers, &storage](int index) 
			{
				splitmix64 rnd;
				rnd.x = k * 100000 + index;
				for( int j = 0; j < nItemPerThread; j++ )
				{
					uint32_t v = rnd.next() % Numbers;
					storage.insert( v );
				}
			} );
		}
		a += storage.m_table[0];
	}
	sw.stop();

	printf( "con / %s insertion %f ms %d\n", typeid( T ).name(), sw.getMs(), a );
}
int main( int argc, char** argv )
{
	// Test
	runTest<LP>();
	runTest<BLP>();
	runTest<BLPZeroEmpty>();
	runTest<BLPZeroEmptyBranchless>();
	runTest<RH>();

	runConcurrentTest<LP_Concurrent>();

	{
		runConcurrentPerfTest<LP_Concurrent>();

		runPerfTest<LP>();
		runPerfTest<RH>();
		runPerfTest<BLP>();
		runPerfTest<BLPZeroEmpty>();
		runPerfTest<BLPZeroEmptyBranchless>();
	}
	
	//for(int i = 0 ; i < 10 ; i++)
	//{
	//	int NBuckets = 1000;
	//	int Numbers = 10000;
	//	double loadFactor = 0.75;

	//	Stopwatch sw;
	//	sw.start();

	//	int nfound = 0;
	//	splitmix64 rnd;
	//	for( int i = 0; i < 100000; i++ )
	//	{
	//		BLPZeroEmptyBranchless lp( NBuckets );
	//		for( int j = 0; j < NBuckets * loadFactor; j++ )
	//		{
	//			uint32_t v = rnd.next() % Numbers;
	//			lp.insert( v );
	//		}

	//		for( int i = 0; i < NBuckets * loadFactor; i++ )
	//		{
	//			uint32_t v = rnd.next() % Numbers;
	//			bool found = lp.find( v ) != -1;
	//			if( found )
	//			{
	//				nfound++;
	//			}
	//		}
	//	}

	//	sw.stop();
	//	printf( "%s %f ms, %d\n", typeid( BLPZeroEmptyBranchless ).name(), sw.getMs(), nfound );
	//}
	return EXIT_SUCCESS;
}
