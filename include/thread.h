#ifndef HAVE_GPTL_THREAD_H
#define HAVE_GPTL_THREAD_H

#include <stdio.h>

#ifdef UNDERLYING_PTHREADS
#define MAX_THREADS 64
#endif

extern volatile int GPTLmax_threads;
extern volatile int GPTLnthreads;
extern int GPTLthreadinit (void);
extern void GPTLthreadfinalize (void);
#ifdef INLINE_THREADING
extern inline int GPTLget_thread_num (void);
#else
extern int GPTLget_thread_num (void);
#endif

extern void GPTLprint_threadmapping (FILE *fp);

#endif
