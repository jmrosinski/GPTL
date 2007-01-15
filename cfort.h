#if ( defined SUNOS ) || ( defined IRIX64 ) || ( defined OSF1 ) || ( defined LINUX ) || ( defined NEC_SX ) || ( defined UNICOSMP )
#define FORTRANUNDERSCORE
#elif ( defined GNUFORTRAN ) || ( defined G95 )
#define FORTRANDOUBLEUNDERSCORE
#endif
