/*
** $Id: util.c,v 1.9 2004-11-03 02:58:25 rosinski Exp $
*/

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "private.h"

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

static bool abort_on_error = false;

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

void GPTset_abort_on_error (bool val)
{
  abort_on_error = val;
}

void *GPTallocate (const int nbytes)
{
  void *ptr;

  if ( ! (ptr = malloc (nbytes)))
    (void) GPTerror ("GPTallocate: malloc failed for %d bytes\n", nbytes);

  return ptr;
}

