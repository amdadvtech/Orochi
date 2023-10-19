#pragma once
#include <Orochi/Orochi.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

inline oroApi getApiType( int argc, char** argv )
{
	oroApi api = ORO_API_HIP;
	if( argc >= 2 )
	{
		if( strcmp( argv[1], "hip" ) == 0 ) api = ORO_API_HIP;
		if( strcmp( argv[1], "cuda" ) == 0 ) api = ORO_API_CUDA;
	}
	return api;
}

inline void checkError( oroError e )
{
	const char* pStr;
	oroGetErrorString( e, &pStr );
	if( e != oroSuccess ) printf( "ERROR==================\n%s\n", pStr );
}

inline void load( const std::filesystem::path filepath, std::vector<char>& dst ) noexcept
{
	std::fstream fin( filepath, std::ios::in );

	if( !fin )
	{
		const auto err = "File Error: " + filepath.string();
		std::cerr << err << std::endl;
		throw std::runtime_error( err );
	}

	fin.seekg( 0, std::ios::end );
	size_t filesize = fin.tellg();
	fin.seekg( 0, std::ios::beg );

	dst.resize( filesize );

	fin.read( dst.data(), filesize );
}

#define ERROR_CHECK( e ) checkError( e )