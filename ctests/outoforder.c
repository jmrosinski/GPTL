#include "../gptl.h"
#include <stdio.h>

/*
** Purpose: test behavior of imperfectly-nested regions
*/
int main ()
{
  int ret;
  int num_errors;
  
  printf ("Testing out of order calls and GPTLnum_errors...\n");
  ret = GPTLinitialize ();
  
  ret = GPTLstart ("zzz");
  ret = GPTLstart ("yyy");
  ret = GPTLstop ("zzz");
  ret = GPTLstop ("yyy");

  ret = GPTLstart ("A");
  ret = GPTLstart ("B");
  ret = GPTLstart ("C");
  ret = GPTLstop ("C");
  ret = GPTLstop ("B");
  ret = GPTLstop ("A");

  ret = GPTLpr (0);
  num_errors = GPTLnum_errors ();
  if (num_errors > 0) {
    printf ("Success: %d errors were found\n", num_errors);
    return 0;
  } else {
    printf ("Failure: no errors were found when they should have been\n"); 
    return 1;
  }
}
