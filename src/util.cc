#include "config.h" // Must be first include
#include "gptl.h"
#include "once.h"
#include "util.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_LIBMPI
#include <mpi.h>
#endif

// Namespace declarations of data/functions to be used only by GPTL
namespace gptl_util {
  int num_warn = 0;               // number of times warn() was called
  int num_errors = 0;             // number of times error() was called
  bool abort_on_error = false;    // abort on error
  extern "C" {
    /*
    ** error: error return routine to print a message and return a failure value.
    **
    ** Input arguments:
    **   fmt: format string
    **   variable list of additional arguments for vfprintf
    **
    ** Return value: -1 (failure)
    */
    int error (const char *fmt, ...)
    {
      va_list args;
    
      if (doprint()) {
	va_start (args, fmt);
	if (fmt != NULL && num_errors < max_errors) {
	  (void) fprintf (stderr, "GPTL error:");
	  (void) vfprintf (stderr, fmt, args);
	  if (num_errors == max_errors)
	    (void) fprintf (stderr, "Truncating further error print now after %d msgs", num_errors);
	}
	va_end (args);
      }
      
      if (abort_on_error) {
	(void) fprintf (stderr, "GPTL error:");
	exit (-1);
      }
      ++num_errors;
      return -1;
    }

    /*
    ** warn: print a warning message
    **
    ** Input arguments:
    **   fmt: format string
    **   variable list of additional arguments for vfprintf
    */
    void warn (const char *fmt, ...)
    {
      va_list args;
      
      if (doprint()) {
	va_start (args, fmt);
	if (fmt != NULL && num_warn < max_warn) {
	  (void) fprintf (stderr, "GPTL warning:");
	  (void) vfprintf (stderr, fmt, args);
	  if (num_warn == max_warn)
	    (void) fprintf (stderr, "Truncating further warning print now after %d msgs", num_warn);
	}
	va_end (args);
      }
      ++num_warn;
    }

    /*
    ** note: print a note
    **
    ** Input arguments:
    **   fmt: format string
    **   variable list of additional arguments for vfprintf
    */
    void note (const char *fmt, ...)
    {
      va_list args;
      
      if (doprint()) {
	va_start (args, fmt);
	if (fmt != NULL) {
	  (void) fprintf (stderr, "GPTL note:");
	  (void) vfprintf (stderr, fmt, args);
	}
	va_end (args);
      }
    }
    
    /*
    ** allocate: wrapper utility for malloc
    **
    ** Input arguments:
    **   nbytes: size to allocate
    **
    ** Return value: pointer to the new space (or NULL)
    */
    void *allocate (const int nbytes, const char *caller)
    {
      void *ptr;
      
      if ( nbytes <= 0 || ! (ptr = malloc (nbytes)))
	(void) error ("GPTLallocate from %s: malloc failed for %d bytes\n", nbytes, caller);
      
      return ptr;
    }
    
#ifdef HAVE_LIBMPI
    /* 
    ** GPTLbarrier: When MPI enabled, set and time an MPI barrier
    **
    ** Input arguments:
    **   comm: commuicator (e.g. MPI_COMM_WORLD). If zero, use MPI_COMM_WORLD
    **   name: region name
    **
    ** Return value: 0 (success)
    */
    int GPTLbarrier (MPI_Comm comm, const char *name)
    {
      static const char *thisfunc = "GPTLbarrier";
      
      int ret;
      
      ret = GPTLstart (name);
      if ((ret = MPI_Barrier (comm)) != MPI_SUCCESS)
	return GPTLerror ("%s: Bad return from MPI_Barrier=%d", thisfunc, ret);
      ret = GPTLstop (name);
      return ret;
    }
#endif    // HAVE_LIBMPI
  }
  
  namespace {
    const int max_errors = 10;   // max number of error print msgs
    const int max_warn = 10;     // max number of warning messages

    extern "C" {
      inline bool doprint()
      {
#ifdef HAVE_LIBMPI
	static int world_iam = 0;  // MPI rank
	int flag;
	int ret;
	static bool check_mpi_init = true;

	// Only need to change GPTLworld_iam from 0 in MPI jobs, when flag set to only print 
	// from rank 0, and when MPI_Initialize hasn't already been invoked
	if (gptl_once::onlypr_rank0 && check_mpi_init) {
	  ret = MPI_Initialized (&flag);
	  if (flag) {
	    check_mpi_init = false;
	    ret = MPI_Comm_rank (MPI_COMM_WORLD, &world_iam);
	  }
	}
	return world_iam == 0;
#else
	return true;  // Always return true when no MPI
#endif
      }
    }
  }
}
