#include <common/Common.h>

void checkOro( oroError res, const char* file, int line )
{
	if( res != oroSuccess )
	{
		const char* msg;
		oroGetErrorString( res, &msg );
		std::cerr << "Orochi error: '" << msg << "' on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void checkOrortc( orortcResult res, const char* file, int line )
{
	if( res != ORORTC_SUCCESS )
	{
		std::cerr << "Orortc error: '" << orortcGetErrorString( res ) << "' [ " << res << " ] on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}
