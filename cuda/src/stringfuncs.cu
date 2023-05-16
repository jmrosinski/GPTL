#include "config.h" // Must be first include.
#include "stringfuncs.h"

namespace stringfuncs {
  __device__ int my_strlen (const char *str)
  {
    const char *s;
    for (s = str; *s; ++s);
    return(s - str);
  }

  __device__ char *my_strcpy (char *dest, const char *src)
  {
    char *ret = dest;
    
    while (*src != '\0')
      *dest++ = *src++;
    *dest = '\0';
    return ret;
  }

  //JR Both of these have about the same performance
  __device__ int my_strcmp (const char *str1, const char *str2)
  {
#ifndef MINE
    while (*str1 == *str2) {
      if (*str1 == '\0')
	break;
      ++str1;
      ++str2;
    }
    return (int) (*str1 - *str2);
#else
    register const unsigned char *s1 = (const unsigned char *) str1;
    register const unsigned char *s2 = (const unsigned char *) str2;
    register unsigned char c1, c2;
    
    do {
      c1 = (unsigned char) *s1++;
      c2 = (unsigned char) *s2++;
      if (c1 == '\0')
	return c1 - c2;
    } while (c1 == c2); 
    return c1 - c2;
#endif
  }
}
