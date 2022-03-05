/*
** Test passing region with too many chars.
*/

#include "config.h"
#include "gptl.h"
#include <stdio.h>

int main ()
{
  int handle;
  int ret;
  int retval;
  double val;

  // ASSUME MAX_CHARS < 64
  char *str = "0123456701234567012345670123456701234567012345670123456701234567";
  ret = GPTLinitialize ();
  ret = GPTLinit_handle (str, &handle);
  ret = GPTLstart (str);
  retval = 0;
  if (ret == 0) {
    printf ("Unexpected success when passing in too many chars, GPTLstart returned %d\n", ret);
    
    retval = 1;
  } else {
    printf ("As expected when passing in too many chars, GPTLstart returned %d\n", ret);
    ret = GPTLstart_handle (str, &handle);
    if (ret == 0) {
      printf ("Unexpected success when passing in too many chars, GPTLstart_handle returned %d\n", ret);
    
      retval = 1;
    }
    return retval;
  }
}
