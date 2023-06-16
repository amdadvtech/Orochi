#pragma once
#include <set>
#include <mutex>
#include <common/Common.h>

// Reference
// [0] Ordered hash tables ( original idea of BLP, very old )
// [1] A Concurrent Bidirectional Linear Probing Algorithm

// Other resource
// https://github.com/senderista/hashtable-benchmarks

// CPU Reference Implementation of Linear Probing
class LP
{
  public:
	enum
	{
		OCCUPIED_BIT = 1 << 31,
		VALUE_MASK = ~OCCUPIED_BIT,
	};

	LP( int n ) : m_table( n ) {}

	int home( uint32_t k ) const { return hash( k ) % m_table.size(); }

	enum InsertionResult
	{
		INSERTED,
		FOUND,
		OUT_OF_MEMORY
	};
	// k must be less than equal 0x7FFFFFFF
	InsertionResult insert( uint32_t k )
	{
		uint32_t h = home( k );
		for( int i = 0; i < m_table.size(); i++ )
		{
			int location = ( h + i ) % m_table.size();

			if( ( m_table[location] & OCCUPIED_BIT ) == 0 )
			{
				m_table[location] = k | OCCUPIED_BIT;
				return INSERTED;
			}
			else if( ( m_table[location] & VALUE_MASK ) == k )
			{
				return FOUND;
			}
		}
		return OUT_OF_MEMORY;
	}
	bool find( uint32_t k ) const
	{
		uint32_t h = home( k );
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

class StdSet_Concurrent
{
  public:
	StdSet_Concurrent( int n = 0 ) {}

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

// CPU Reference Implementation of Bidirectional Linear Probing
class BLP
{
  public:
	enum
	{
		OCCUPIED_BIT = 1 << 31,
		VALUE_MASK = ~OCCUPIED_BIT,
	};

	BLP( int n ) : m_table( n ) {}

	uint32_t home( uint32_t k ) const
	{
		uint32_t v = hash( k );
		return (uint64_t)v * m_table.size() / ( (uint64_t)UINT_MAX + 1 );
	}
	enum InsertionResult
	{
		INSERTED,
		FOUND,
		OUT_OF_MEMORY
	};
	// k must be less than equal 0x7FFFFFFF
	InsertionResult insert( uint32_t k )
	{
		uint32_t h = home( k );
		if( ( m_table[h] & OCCUPIED_BIT ) == 0 )
		{
			m_table[h] = k | OCCUPIED_BIT;
			return INSERTED;
		}

		if( m_table[h] == ( k | OCCUPIED_BIT ) )
		{
			return FOUND;
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
						return FOUND;
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
						return FOUND;
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
			return INSERTED;
		}
		return OUT_OF_MEMORY;
	}
	bool find( uint32_t k ) const
	{
		uint32_t h = home( k );
		if( ( m_table[h] & OCCUPIED_BIT ) == 0 )
		{
			return false;
		}
		if( m_table[h] == ( k | OCCUPIED_BIT ) )
		{
			return true;
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
				return true;
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
				return true;
			}
		}
		return false;
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