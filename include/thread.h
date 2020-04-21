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
extern
#ifdef INLINE_THREADING
inline
#endif
int GPTLget_thread_num (void);
extern void GPTLprint_threadmapping (FILE *fp);

#endif
