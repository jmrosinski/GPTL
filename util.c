/*
** $Id: util.c,v 1.3 2004-10-14 19:25:55 rosinski Exp $
*/

#include <stdarg.h>
#include <stdio.h>

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
    (void) vfprintf (stderr, fmt, args);

  va_end (args);
  
  return (-1);
}
