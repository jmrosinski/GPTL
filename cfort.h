#if ( defined GFORTRAN ) || ( defined G95 ) || ( defined LINUX )
#define FORTRANDOUBLEUNDERSCORE
#elif ( defined SUNOS ) || ( defined IRIX64 ) || ( defined OSF1 ) || ( defined NEC_SX ) || ( defined UNICOSMP )
#define FORTRANUNDERSCORE
#endif
