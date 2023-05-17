#ifndef GPTLSTRINGFUNCS_H
#define GPTLSTRINGFUNCS_H

#define STRMATCH(X,Y) (stringfuncs::my_strcmp((X),(Y)) == 0)

namespace stringfuncs {
  extern __device__ int my_strlen (const char *);
  extern __device__ char *my_strcpy (char *, const char *);
  extern __device__ int my_strcmp (const char *, const char *);
}
#endif
