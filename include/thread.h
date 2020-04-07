#ifndef HAVE_GPTL_THREAD_H
#define HAVE_GPTL_THREAD_H

#include "config.h" // Must be first include
#include <stdio.h>

namespace gptl_thread {
  extern volatile int maxthreads;
  extern volatile int nthreads;

  extern "C" {
    int threadinit (void);
    void threadfinalize (void);
    inline int get_thread_num (void);
    void print_threadmapping (FILE *fp);
  }
}
#endif
