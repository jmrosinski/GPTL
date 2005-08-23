/*
** $Id: f_wrappers.c,v 1.11 2005-08-23 02:21:27 rosinski Exp $
** 
** Fortran wrappers for timing library routines
*/

#include <string.h>
#include "private.h"

#if ( defined FORTRANCAPS )

#define gptlinitialize GPTLINITIALIZE
#define gptlfinalize GPTLFINALIZE
#define gptlpr GPTLPR
#define gptlreset GPTLRESET
#define gptlstamp GPTLSTAMP
#define gptlstart GPTLSTART
#define gptlstop GPTLSTOP
#define gptlsetoption GPTLSETOPTION
#define gptlpapiprinttable GPTLPAPIPRINTTABLE

#elif ( defined FORTRANUNDERSCORE )

#define gptlinitialize gptlinitialize_
#define gptlfinalize gptlfinalize_
#define gptlpr gptlpr_
#define gptlreset gptlreset_
#define gptlstamp gptlstamp_
#define gptlstart gptlstart_
#define gptlstop gptlstop_
#define gptlsetoption gptlsetoption_
#define gptlpapiprinttable gptlpapiprinttable_

#elif ( defined FORTRANDOUBLEUNDERSCORE )

#define gptlinitialize gptlinitialize__
#define gptlfinalize gptlfinalize__
#define gptlpr gptlpr__
#define gptlreset gptlreset__
#define gptlstamp gptlstamp__
#define gptlstart gptlstart__
#define gptlstop gptlstop__
#define gptlsetoption gptlsetoption__
#define gptlpapiprinttable gptlpapiprinttable__

#endif

#ifdef NUMERIC_TIMERS
int gptlstart (unsigned long);
int gptlstop (unsigned long);
#else
int gptlstart (char *, int);
int gptlstop (char *, int);
#endif

int gptlinitialize ()
{
  return GPTLinitialize ();
}

int gptlfinalize ()
{
  return GPTLfinalize ();
}

int gptlpr (int *procid)
{
  return GPTLpr (*procid);
}

void gptlreset ()
{
  GPTLreset();
  return;
}

int gptlstamp (double *wall, double *usr, double *sys)
{
  return GPTLstamp (wall, usr, sys);
}

#ifdef NUMERIC_TIMERS

int gptlstart (unsigned long tag)
{
  return GPTLstart (tag);
}

int gptlstop (unsigned long tag)
{
  return GPTLstop (tag);
}

#else

int gptlstart (char *name, int nc1)
{
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (nc1, MAX_CHARS);
  strncpy (cname, name, numchars);
  cname[numchars] = '\0';
  return GPTLstart (cname);
}

int gptlstop (char *name, int nc1)
{
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (nc1, MAX_CHARS);
  strncpy (cname, name, numchars);
  cname[numchars] = '\0';
  return GPTLstop (cname);
}

#endif

int gptlsetoption (int *option, int *val)
{
  return GPTLsetoption (*option, (bool) *val);
}

#ifdef HAVE_PAPI
void gptlpapiprinttable ()
{
  GPTLPAPIprinttable ();
  return;
}
#endif
