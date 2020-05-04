/* 
** Check memusage GPTL capability
*/

#include "gptl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int onemb = 1024 * 1024;
void sub (unsigned char *, int);

int main ()
{
  int ret;
  int n;
  unsigned char *arr = NULL;

  // Print when process size has grown.
  if ((ret = GPTLsetoption (GPTLdopr_memusage, 1)) != 0)
    return -1;
  
  // Only print when the process has grown by 50% or more since the last print
  // (or since the process started)
  if ((ret = GPTLsetoption (GPTLmem_growth, 50)) != 0)
    return -1;
  
  ret = GPTLinitialize ();
  for (n = 1; n < 10; ++n)
    sub (arr, n);
  return 0;
}

void sub (unsigned char *arr, int n)
{
  unsigned char *space;
  int ret;

  ret = GPTLstart ("sub");
  space = (unsigned char *) realloc (arr, n*onemb*(sizeof (unsigned char)));
  arr = space;
  memset (arr, 0, n*onemb*(sizeof (unsigned char)));
  ret = GPTLstop ("sub");
}
