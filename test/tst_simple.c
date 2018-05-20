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
   printf("*** testing initialization/finalization...");
   {
      if (GPTLinitialize()) ERR;
      if (GPTLfinalize()) ERR;
   }
   printf("\n*** SUCCESS!\n");
   return 0;
}

