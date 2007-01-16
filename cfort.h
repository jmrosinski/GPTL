#if ( defined GFORTRAN ) || ( defined G95 )
#define FORTRANDOUBLEUNDERSCORE
#elif ( defined SUNOS ) || ( defined IRIX64 ) || ( defined OSF1 ) || ( defined LINUX ) || ( defined NEC_SX ) || ( defined UNICOSMP )
#define FORTRANUNDERSCORE
#endif
