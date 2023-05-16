#define STRMATCH(X,Y) (my_strcmp((X),(Y)) == 0)

namespace stringfuncs {
  extern __device__ {
    int my_strlen (const char *);
    char *my_strcpy (char *, const char *);
    int my_strcmp (const char *, const char *);
  }
}
