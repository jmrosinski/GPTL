/*
** $Id: f_wrappers.c,v 1.2 2001-01-01 19:34:05 rosinski Exp $
** 
** Fortran wrappers for timing library routines
*/

#if ( defined CRAY ) || ( defined T3D )
#include <fortran.h>
#endif

#include "gpt.h"
#include "private.h"

#if ( defined FORTRANCAPS )

#define gptinitialize GPTINITIALIZE
#define gptpr GPTPR
#define gptreset GPTRESET
#define gptstamp GPTSTAMP
#define gptstart GPTSTART
#define gptstop GPTSTOP
#define gptsetoption GPTSETOPTION

#elif ( defined FORTRANUNDERSCORE )

#define gptinitialize gptinitialize_
#define gptpr gptpr_
#define gptreset gptreset_
#define gptstamp gptstamp_
#define gptstart gptstart_
#define gptstop gptstop_
#define gptsetoption gptsetoption_

#elif ( defined FORTRANDOUBLEUNDERSCORE )

#define gptinitialize gptinitialize__
#define gptpr gptpr__
#define gptreset gptreset__
#define gptstamp gptstamp__
#define gptstart gptstart__
#define gptstop gptstop__
#define gptsetoption gptsetoption__

#endif

#if ( defined CRAY ) || ( defined T3D )

int gptstart (_fcd);
int gptstop (_fcd);

#else

int gptstart (char *, int);
int gptstop (char *, int);

#endif

int gptinitialize ()
{
  return GPTinitialize ();
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
  return GPTsetoption ( (OptionName) *option, (Boolean) *val);
}

int gptstamp (double *wall, double *usr, double *sys)
{
  return GPTstamp (wall, usr, sys);
}

#if ( defined CRAY ) || ( defined T3D )

int gptstart (_fcd name)
{
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (_fcdlen (name), MAX_CHARS);
  strncpy (cname, _fcdtocp (name), numchars);
  cname[numchars] = '\0';
  return GPTstart (cname);
}

int gptstop (_fcd name)
{
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (_fcdlen (name), MAX_CHARS);
  strncpy (cname, _fcdtocp (name), numchars);
  cname[numchars] = '\0';
  return GPTstop (cname);
}

#else

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

#endif
