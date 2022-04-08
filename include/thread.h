#ifndef HAVE_GPTL_THREAD_H
#define HAVE_GPTL_THREAD_H

#include <stdio.h>

#ifdef UNDERLYING_PTHREADS
#define MAX_THREADS 64
#endif

namespace thread {
  extern volatile int max_threads;
  extern volatile int nthreads;
  extern volatile int *threadid;

  extern "C" {
    int threadinit (void);
    void threadfinalize (void);
#ifdef INLINE_THREADING
    inline int get_thread_num (void);
#else
    int get_thread_num (void);
#endif
    void print_threadmapping (FILE *fp);
  }
}
#endif
