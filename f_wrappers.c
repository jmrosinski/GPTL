/*
** $Id: f_wrappers.c,v 1.1 2000-12-27 06:10:31 rosinski Exp $
** 
** Fortran wrappers for timing library routines
*/

#if ( defined CRAY ) || ( defined T3D )
#include <fortran.h>
#endif

#include "header.h"

#if ( defined FORTRANCAPS )

#define GPTinitialize GPTINITIALIZE
#define GPTpr GPTPR
#define GPTreset GPTRESET
#define GPTstamp GPTSTAMP
#define GPTstart GPTSTART
#define GPTstop GPTSTOP
#define GPTsetoption GPTSETOPTION

#elif ( defined FORTRANUNDERSCORE )

#define GPTinitialize GPTinitialize_
#define GPTpr GPTpr_
#define GPTreset GPTreset_
#define GPTstamp GPTstamp_
#define GPTstart GPTstart_
#define GPTstop GPTstop_
#define GPTsetoption GPTsetoption_

#elif ( defined FORTRANDOUBLEUNDERSCORE )

#define GPTinitialize GPTinitialize__
#define GPTpr GPTpr__
#define GPTreset GPTreset__
#define GPTstamp GPTstamp__
#define GPTstart GPTstart__
#define GPTstop GPTstop__
#define GPTsetoption GPTsetoption__

#endif

#if ( defined CRAY ) || ( defined T3D )

int GPTstart (_fcd);
int GPTstop (_fcd);

#else

int GPTstart (char *, int);
int GPTstop (char *, int);

#endif

int GPTinitialize ()
{
  return GPTinitialize ();
}

int GPTpr (int *procid)
{
  return GPTpr (*procid);
}

void GPTreset ()
{
  GPTreset();
  return;
}

int GPTsetoption (int *option, int *val)
{
  return GPTsetoption ( (OptionName) *option, (Boolean) *val);
}

int GPTstamp (double *wall, double *usr, double *sys)
{
  return GPTstamp (wall, usr, sys);
}

#if ( defined CRAY ) || ( defined T3D )

int GPTstart (_fcd name)
{
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (_fcdlen (name), MAX_CHARS);
  strncpy (cname, _fcdtocp (name), numchars);
  cname[numchars] = '\0';
  return GPTstart (cname);
}

int GPTstop (_fcd name)
{
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (_fcdlen (name), MAX_CHARS);
  strncpy (cname, _fcdtocp (name), numchars);
  cname[numchars] = '\0';
  return GPTstop (cname);
}

#else

int GPTstart (char *name, int nc1)
{
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (nc1, MAX_CHARS);
  strncpy (cname, name, numchars);
  cname[numchars] = '\0';
  return GPTstart (cname);
}

int GPTstop (char *name, int nc1)
{
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (nc1, MAX_CHARS);
  strncpy (cname, name, numchars);
  cname[numchars] = '\0';
  return GPTstop (cname);
}

#endif
