#include "private.h"
#include <stdio.h>

namespace memusage {
  extern float growth_pct;
  extern "C" {
    void check_memusage (const char *, const char *);
    void print_memstats (FILE *, Timer **, int);
  }
}
