/*
** $Id: f_wrappers.c,v 1.7 2004-10-19 03:16:18 rosinski Exp $
** 
** Fortran wrappers for timing library routines
*/

#include <string.h>
#include "private.h"

#if ( defined FORTRANCAPS )

#define gptinitialize GPTINITIALIZE
#define gptfinalize GPTFINALIZE
#define gptpr GPTPR
#define gptreset GPTRESET
#define gptstamp GPTSTAMP
#define gptstart GPTSTART
#define gptstop GPTSTOP
#define gptsetoption GPTSETOPTION

#elif ( defined FORTRANUNDERSCORE )

#define gptinitialize gptinitialize_
#define gptfinalize gptfinalize_
#define gptpr gptpr_
#define gptreset gptreset_
#define gptstamp gptstamp_
#define gptstart gptstart_
#define gptstop gptstop_
#define gptsetoption gptsetoption_

#elif ( defined FORTRANDOUBLEUNDERSCORE )

#define gptinitialize gptinitialize__
#define gptfinalize gptfinalize__
#define gptpr gptpr__
#define gptreset gptreset__
#define gptstamp gptstamp__
#define gptstart gptstart__
#define gptstop gptstop__
#define gptsetoption gptsetoption__

#endif

int gptstart (char *, int);
int gptstop (char *, int);

int gptinitialize ()
{
  return GPTinitialize ();
}

int gptfinalize ()
{
  return GPTfinalize ();
}

int gptpr (int *procid)
{
  return GPTpr (*procid);
}

void gptreset ()
{
  GPTreset();
  return;
}

int gptsetoption (int *option, int *val)
{
  return GPTsetoption ((GPTOption) *option, (bool) *val);
}

int gptstamp (double *wall, double *usr, double *sys)
{
  return GPTstamp (wall, usr, sys);
}

int gptstart (char *name, int nc1)
{
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (nc1, MAX_CHARS);
  strncpy (cname, name, numchars);
  cname[numchars] = '\0';
  return GPTstart (cname);
}

int gptstop (char *name, int nc1)
{
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (nc1, MAX_CHARS);
  strncpy (cname, name, numchars);
  cname[numchars] = '\0';
  return GPTstop (cname);
}
