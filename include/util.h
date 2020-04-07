#ifndef UTIL_H
#define UTIL_H

namespace gptl_util {
  extern int num_warn;
  extern int num_errors;
  extern bool abort_on_error;
  extern "C" {
    int error (const char *, ...);
    void warn (const char *, ...);
    void note (const char *, ...);
    void *allocate (const int, const char *);
#ifdef HAVE_LIBMPI
    int GPTLbarrier (MPI_Comm, const char *);
#endif
  }
  
  namespace {
    extern const int max_errors;
    extern const int max_warn;
    extern "C" {
      inline bool doprint(void);
    }
  }
}
#endif
