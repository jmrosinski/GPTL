/*
** $Id: util.c,v 1.1 2000-12-27 06:10:31 rosinski Exp $
*/

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "gpt.h"

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

char *GPTpclstr (int code)
{
#ifdef HAVE_PCL
  switch (code) {

  case PCL_SUCCESS: 
    return "Success";
    
  case PCL_NOT_SUPPORTED:
    return "Event not supported";
    
  case PCL_TOO_MANY_EVENTS:
    return "Too many events";
    
  case PCL_TOO_MANY_NESTINGS:
    return "More nesting levels than allowed";
    
  case PCL_ILL_NESTING:
    return "Bad nesting";
    
  case PCL_ILL_EVENT:
    return "Illegal event identifier";
    
  case PCL_MODE_NOT_SUPPORTED:
    return "Mode not supported";
    
  case PCL_FAILURE:
    return "Failure for unspecified reason";
    
  default:
    return "Unknown error code";
    
  }
#endif
}

      
  
