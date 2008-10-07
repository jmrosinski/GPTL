/*
** $Id: f_wrappers.c,v 1.34 2008-10-07 19:34:00 rosinski Exp $
** 
** Fortran wrappers for timing library routines
*/

#include <string.h>
#include <stdlib.h>
#include "private.h" /* MAX_CHARS, bool */
#include "gptl.h"    /* function prototypes */

#if ( defined FORTRANCAPS )

#define gptlinitialize GPTLINITIALIZE
#define gptlfinalize GPTLFINALIZE
#define gptlpr GPTLPR
#define gptlpr_file GPTLPR_FILE
#define gptlpr_summary GPTLPR_SUMMARY
#define gptlreset GPTLRESET
#define gptlstamp GPTLSTAMP
#define gptlstart GPTLSTART
#define gptlstop GPTLSTOP
#define gptlsetoption GPTLSETOPTION
#define gptlenable GPTLENABLE
#define gptldisable GPTLDISABLE
#define gptlsetutr GPTLSETUTR
#define gptlquery GPTLQUERY
#define gptlquerycounters GPTLQUERYCOUNTERS
#define gptlget_nregions GPTLGET_NREGIONS
#define gptlget_regionname GPTLGET_REGIONNAME
#define gptlget_memusage GPTLGET_MEMUSAGE
#define gptlprint_memusage GPTLPRINT_MEMUSAGE
#define gptl_papilibraryinit GPTL_PAPILIBRARYINIT
#define gptlevent_name_to_code GPTLEVENT_NAME_TO_CODE
#define gptlevent_code_to_name GPTLEVENT_CODE_TO_NAME

#elif ( defined FORTRANUNDERSCORE )

#define gptlinitialize gptlinitialize_
#define gptlfinalize gptlfinalize_
#define gptlpr gptlpr_
#define gptlpr_file gptlpr_file_
#define gptlpr_summary gptlpr_summary_
#define gptlreset gptlreset_
#define gptlstamp gptlstamp_
#define gptlstart gptlstart_
#define gptlstop gptlstop_
#define gptlsetoption gptlsetoption_
#define gptlenable gptlenable_
#define gptldisable gptldisable_
#define gptlsetutr gptlsetutr_
#define gptlquery gptlquery_
#define gptlquerycounters gptlquerycounters_
#define gptlget_nregions gptlget_nregions_
#define gptlget_regionname gptlget_regionname_
#define gptlget_memusage gptlget_memusage_
#define gptlprint_memusage gptlprint_memusage_
#define gptl_papilibraryinit gptl_papilibraryinit_
#define gptlevent_name_to_code gptlevent_name_to_code_
#define gptlevent_code_to_name gptlevent_code_to_name_

#elif ( defined FORTRANDOUBLEUNDERSCORE )

#define gptlinitialize gptlinitialize_
#define gptlfinalize gptlfinalize_
#define gptlpr gptlpr_
#define gptlpr_file gptlpr_file__
#define gptlpr_summary gptlpr_summary__
#define gptlreset gptlreset_
#define gptlstamp gptlstamp_
#define gptlstart gptlstart_
#define gptlstop gptlstop_
#define gptlsetoption gptlsetoption_
#define gptlenable gptlenable_
#define gptldisable gptldisable_
#define gptlsetutr gptlsetutr_
#define gptlquery gptlquery_
#define gptlquerycounters gptlquerycounters_
#define gptlget_nregions gptlget_nregions__
#define gptlget_regionname gptlget_regionname__
#define gptlget_memusage gptlget_memusage__
#define gptlprint_memusage gptlprint_memusage__
#define gptl_papilibraryinit gptl_papilibraryinit__
#define gptlevent_name_to_code gptlevent_name_to_code__
#define gptlevent_code_to_name gptlevent_code_to_name__

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

int gptlpr_file (char *file, int nc1)
{
  char *locfile;
  int ret;

  if ( ! (locfile = malloc (nc1+1)))
    return GPTLerror ("gptlpr_file: malloc error\n");

  snprintf (locfile, nc1, "%s", file);
  locfile[nc1] = '\0';

  ret = GPTLpr_file (locfile);
  free (locfile);
  return ret;
}

int gptlpr_summary (int *comm)
{
  return GPTLpr_summary (*comm);
}

int gptlreset ()
{
  return GPTLreset();
}

int gptlstamp (double *wall, double *usr, double *sys)
{
  return GPTLstamp (wall, usr, sys);
}

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

int gptlsetoption (int *option, int *val)
{
  return GPTLsetoption (*option, *val);
}

int gptlenable ()
{
  return GPTLenable ();
}

int gptldisable ()
{
  return GPTLdisable ();
}

int gptlsetutr (int *option)
{
  return GPTLsetutr (*option);
}

int gptlquery (const char *name, int *t, int *count, int *onflg, double *wallclock, 
	       double *usr, double *sys, long long *papicounters_out, int *maxcounters, 
	       int nc)
{
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (nc, MAX_CHARS);
  strncpy (cname, name, numchars);
  cname[numchars] = '\0';
  return GPTLquery (cname, *t, count, onflg, wallclock, usr, sys, papicounters_out, *maxcounters);
}

int gptlquerycounters (const char *name, int *t, long long *papicounters_out, int nc)
{
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (nc, MAX_CHARS);
  strncpy (cname, name, numchars);
  cname[numchars] = '\0';
  return GPTLquerycounters (cname, *t, papicounters_out);
}

int gptlget_nregions (int *t, int *nregions)
{
  return GPTLget_nregions (*t, nregions);
}

int gptlget_regionname (int *t, int *region, char *name, int nc)
{
  return GPTLget_regionname (*t, *region, name, nc);
}

int gptlget_memusage (int *size, int *rss, int *share, int *text, int *datastack)
{
  return GPTLget_memusage (size, rss, share, text, datastack);
}

int gptlprint_memusage (const char *str, int nc)
{
  char cname[128+1];
  int numchars = MIN (nc, 128);

  strncpy (cname, str, numchars);
  cname[numchars] = '\0';
  return GPTLprint_memusage (cname);
}

#ifdef HAVE_PAPI
#include <papi.h>

void gptl_papilibraryinit ()
{
  (void) GPTL_PAPIlibraryinit ();
  return;
}

int gptlevent_name_to_code (const char *str, int *code, int nc)
{
  char cname[PAPI_MAX_STR_LEN];
  int numchars = MIN (nc, PAPI_MAX_STR_LEN);

  strncpy (cname, str, numchars);
  cname[numchars] = '\0';

  /* "code" is an int* and is an output variable */

  return GPTLevent_name_to_code (cname, code);
}

int gptlevent_code_to_name (int *code, char *str, int nc)
{
  int i;

  if (nc < PAPI_MAX_STR_LEN)
    return GPTLerror ("gptl_event_code_to_name: output name must hold at least %d characters\n",
		      PAPI_MAX_STR_LEN);

  if (GPTLevent_code_to_name (*code, str) == 0) {
    for (i = strlen(str); i < nc; ++i)
      str[i] = ' ';
  } else {
    return GPTLerror ("gptl_event_code_to_name\n");
  }
  return 0;
}
#endif
