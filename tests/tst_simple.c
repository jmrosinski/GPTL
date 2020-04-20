/* Simple test for GPTL. 
 *
 * Ed Hartnett 5/20/18
 */

#include "config.h"
#include "gptl.h"
#include <stdio.h>

/* This macro prints an error message with line number and name of
 * test program. */
#define ERR do { \
fflush(stdout); /* Make sure our stdout is synced with stderr. */ \
fprintf(stderr, "Sorry! Unexpected result, %s, line: %d\n", \
	__FILE__, __LINE__);				    \
fflush(stderr);                                             \
return 2;                                                   \
} while (0)


int
main(int argc, char **argv)
{
   printf("\n*** Testing GPTL.\n");
   printf("*** testing setting options and initialization/finalization...");
   {
      if (GPTLsetoption(GPTLverbose, 1)) ERR;
      if (GPTLsetoption(GPTLwall, 1)) ERR;
      if (GPTLsetoption(GPTLoverhead, 1)) ERR;
      if (GPTLsetoption(GPTLdepthlimit, 1)) ERR;
      if (GPTLsetoption(GPTLpercent, 1)) ERR;
      if (GPTLsetoption(GPTLdopr_preamble, 1)) ERR;
      if (GPTLsetoption(GPTLdopr_threadsort, 1)) ERR;
      if (GPTLsetoption(GPTLdopr_multparent, 1)) ERR;
      if (GPTLsetoption(GPTLdopr_collision, 1)) ERR;
      if (GPTLsetoption(GPTLdopr_memusage, 1)) ERR;
      if (GPTLsetoption(GPTLprint_method, 1)) ERR;
      if (GPTLsetoption(GPTLtablesize, 2)) ERR;
      if (GPTLsetoption(GPTLsync_mpi, 1)) ERR;
      if (GPTLsetoption(GPTLmaxthreads, 1)) ERR;
#ifdef HAVE_PAPI
      if (GPTLsetoption(GPTLmultiplex, 1)) ERR;
#else
      if (GPTLsetoption(GPTLmultiplex, 1) != -1) ERR;
#endif /* HAVE_PAPI */
      if (GPTLinitialize()) ERR;

      /* This will not work. */
      if (GPTLsetoption(GPTLwall, 0) != -1) ERR;
      
      if (GPTLfinalize()) ERR;
   }
   printf("\n*** SUCCESS!\n");
   return 0;
}

