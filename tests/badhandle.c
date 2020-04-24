/*
** Test correct behavior of start/stop routines when passed a bad handle
** Use simple timer names 1 and 2 so timers guaranteed (?) different hash indices
** If the indices for 1 and 2 happened to match, could get a false positive
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


int main ()
{
  int handle;
  int ret;

  ret = GPTLinitialize ();
  ret = GPTLinit_handle ("1", &handle);
  // Correct
  if ((ret = GPTLstart_handle ("1", &handle)) != 0)
    ERR;
  // Error: handle doesn't match what should be generated for name
  if ((ret = GPTLstart_handle ("2", &handle)) == 0)
    ERR;
  // Correct
  if ((ret = GPTLstop_handle ("1", &handle)) != 0)
    ERR;
  // Error: handle doesn't match what should be generated for name
  if ((ret = GPTLstop_handle ("2", &handle)) == 0)
    ERR;
  return 0;
}
  
