/*
** $Id: util.c,v 1.10 2004-11-10 23:55:57 rosinski Exp $
*/

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "private.h"

static bool abort_on_error = false; /* flag says to abort on any error */

/*
** GPTerror: error return routine to print a message and return a failure
** value.
**
** Input arguments:
**   fmt: format string
**   variable list of additional arguments for vfprintf
**
** Return value: -1 (failure)
*/

int GPTerror (const char *fmt, ...)
{
  va_list args;
  
  va_start (args, fmt);
  
  if (fmt != NULL)
#ifdef HAVE_VPRINTF
    (void) vfprintf (stderr, fmt, args);
#else
    (void) fprintf (stderr, "GPTerror: no vfprintf: fmt is %s\n", fmt);
#endif
  
  va_end (args);
  
  if (abort_on_error)
    exit (-1);

  return (-1);
}

/*
** GPTset_abort_on_error: User-visible routine to set abort_on_error flag
**
** Input arguments:
**   val: true (abort on error) or false (don't)
*/

void GPTset_abort_on_error (bool val)
{
  abort_on_error = val;
}

/*
** GPTallocate: wrapper utility for malloc
**
** Input arguments:
**   nbytes: size to allocate
**
** Return value: pointer to the new space (or NULL)
*/

void *GPTallocate (const int nbytes)
{
  void *ptr;

  if ( ! (ptr = malloc (nbytes)))
    (void) GPTerror ("GPTallocate: malloc failed for %d bytes\n", nbytes);

  return ptr;
}

