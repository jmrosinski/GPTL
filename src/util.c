/*
** $Id: util.c,v 1.13 2010-01-01 01:34:07 rosinski Exp $
*/

#include "config.h" /* Must be first include. */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "private.h"
#include "gptl.h"

static bool abort_on_error = false;  /* flag says to abort on any error */
static int max_errors = 10;          /* max number of error print msgs */
static int num_errors = 0;           /* number of times GPTLerror was called */
static int max_warn = 10;            /* max number of warning messages */
static int num_warn = 0;             /* number of times GPTLwarn was called */

/*
** GPTLerror: error return routine to print a message and return a failure
** value.
**
** Input arguments:
**   fmt: format string
**   variable list of additional arguments for vfprintf
**
** Return value: -1 (failure)
*/
int GPTLerror (const char *fmt, ...)
{
  va_list args;
  
  va_start (args, fmt);
  
  if (fmt != NULL && num_errors < max_errors) {
    (void) fprintf (stderr, "GPTL error:");
    (void) vfprintf (stderr, fmt, args);
    if (num_errors == max_errors)
      (void) fprintf (stderr, "Truncating further error print now after %d msgs",
		      num_errors);
  }
  
  va_end (args);
  
  if (abort_on_error)
    exit (-1);

  ++num_errors;
  return (-1);
}

/*
** GPTLwarn: print a warning message
** value.
**
** Input arguments:
**   fmt: format string
**   variable list of additional arguments for vfprintf
*/
void GPTLwarn (const char *fmt, ...)
{
  va_list args;
  
  va_start (args, fmt);
  
  if (fmt != NULL && num_warn < max_warn) {
    (void) fprintf (stderr, "GPTL warning:");
    (void) vfprintf (stderr, fmt, args);
    if (num_warn == max_warn)
      (void) fprintf (stderr, "Truncating further warning print now after %d msgs",
		      num_warn);
  }
  
  va_end (args);
  
  ++num_warn;
}

/*
** GPTLnote: print a note
**
** Input arguments:
**   fmt: format string
**   variable list of additional arguments for vfprintf
*/
void GPTLnote (const char *fmt, ...)
{
  va_list args;
  
  va_start (args, fmt);
  
  if (fmt != NULL) {
    (void) fprintf (stderr, "GPTL note:");
    (void) vfprintf (stderr, fmt, args);
  }
  
  va_end (args);
}

/*
** GPTLset_abort_on_error: User-visible routine to set abort_on_error flag
**
** Input arguments:
**   val: true (abort on error) or false (don't)
*/
void GPTLset_abort_on_error (bool val)
{
  abort_on_error = val;
}

/*
** GPTLreset_errors: reset error state to no errors
**
*/
void GPTLreset_errors (void)
{
  num_errors = 0;
}

/*
** GPTLnum_errors: User-visible routine returns number of times GPTLerror() called
**
*/
int GPTLnum_errors (void)
{
  return num_errors;
}

/*
** GPTLnum_errors: User-visible routine returns number of times GPTLerror() called
**
*/
int GPTLnum_warn (void)
{
  return num_warn;
}

/*
** GPTLallocate: wrapper utility for malloc
**
** Input arguments:
**   nbytes: size to allocate
**
** Return value: pointer to the new space (or NULL)
*/
void *GPTLallocate (const int nbytes, const char *caller)
{
  void *ptr;

  if ( nbytes <= 0 || ! (ptr = malloc (nbytes)))
    (void) GPTLerror ("GPTLallocate from %s: malloc failed for %d bytes\n", nbytes, caller);

  return ptr;
}

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

#ifdef HAVE_LIBMPI
  int ret;

  ret = GPTLstart (name);
  if ((ret = MPI_Barrier (comm)) != MPI_SUCCESS)
    return GPTLerror ("%s: Bad return from MPI_Barrier=%d", thisfunc, ret);
  ret = GPTLstop (name);
  return ret;
#else
  return GPTLerror ("%s: Need to build GPTL with #define HAVE_LIBMPI\n", thisfunc);
#endif    /* HAVE_LIBMPI */
}
