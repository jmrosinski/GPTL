/*
** $Id: util.c,v 1.4 2004-10-15 04:56:27 rosinski Exp $
*/

#include <stdarg.h>
#include <stdio.h>

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

static bool abort_on_error = 0;

int GPTerror (const char *fmt, ...)
{
  va_list args;

  va_start (args, fmt);

  if (fmt != NULL)
    (void) vfprintf (stderr, fmt, args);

  va_end (args);
  
  if (abort_on_error)
    exit (-1);

  return (-1);
}

void GPTset_abort_on_error (bool val)
{
  abort_on_error = val;
}
