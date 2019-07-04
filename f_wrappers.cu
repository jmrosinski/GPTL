/*
** f_wrappers.c
**
** Author: Jim Rosinski
** 
** Fortran wrappers for timing library routines
*/

#ifdef HAVE_LIBMPI
#include <mpi.h>
#endif

#include <string.h>
#include <stdlib.h>
#include "private.h" /* MAX_CHARS, bool */
#include "gptl.h"    /* function prototypes */

#if ( defined FORTRANUNDERSCORE )

#define gptlinitialize gptlinitialize_
#define gptlfinalize gptlfinalize_
#define gptlpr gptlpr_
#define gptlpr_file gptlpr_file_
#define gptlpr_summary gptlpr_summary_
#define gptlpr_summary_file gptlpr_summary_file_
#define gptlbarrier gptlbarrier_
#define gptlreset gptlreset_
#define gptlreset_timer gptlreset_timer_
#define gptlstamp gptlstamp_
#define gptlstart gptlstart_
#define gptlinit_handle gptlinit_handle_
#define gptlstart_handle gptlstart_handle_
#define gptlstop gptlstop_
#define gptlstop_handle gptlstop_handle_
#define gptlsetoption gptlsetoption_
#define gptlenable gptlenable_
#define gptldisable gptldisable_
#define gptlsetutr gptlsetutr_
#define gptlquery gptlquery_
#define gptlquerycounters gptlquerycounters_
#define gptlget_wallclock gptlget_wallclock_
#define gptlget_wallclock_latest gptlget_wallclock_latest_
#define gptlget_threadwork gptlget_threadwork_
#define gptlstartstop_val gptlstartstop_val_
#define gptlget_eventvalue gptlget_eventvalue_
#define gptlget_nregions gptlget_nregions_
#define gptlget_regionname gptlget_regionname_
#define gptlget_memusage gptlget_memusage_
#define gptlprint_memusage gptlprint_memusage_
#define gptlprint_rusage gptlprint_rusage_
#define gptlnum_errors gptlnum_errors_
#define gptlnum_warn gptlnum_warn_
#define gptlget_count gptlget_count_
#define gptl_papilibraryinit gptl_papilibraryinit_
#define gptlevent_name_to_code gptlevent_name_to_code_
#define gptlevent_code_to_name gptlevent_code_to_name_
#define gptlcompute_chunksize gptlcompute_chunksize_
#define gptlget_gpu_props gptlget_gpu_props_
#define gptlcudadevsync gptlcudadevsync_

#elif ( defined FORTRANDOUBLEUNDERSCORE )

#define gptlinitialize gptlinitialize_
#define gptlfinalize gptlfinalize_
#define gptlpr gptlpr_
#define gptlpr_file gptlpr_file__
#define gptlpr_summary gptlpr_summary__
#define gptlpr_summary_file gptlpr_summary_file__
#define gptlbarrier gptlbarrier_
#define gptlreset gptlreset_
#define gptlreset_timer gptlreset_timer__
#define gptlstamp gptlstamp_
#define gptlstart gptlstart_
#define gptlinit_handle gptlinit_handle__
#define gptlstart_handle gptlstart_handle__
#define gptlstop gptlstop_
#define gptlstop_handle gptlstop_handle__
#define gptlsetoption gptlsetoption_
#define gptlenable gptlenable_
#define gptldisable gptldisable_
#define gptlsetutr gptlsetutr_
#define gptlquery gptlquery_
#define gptlquerycounters gptlquerycounters_
#define gptlget_wallclock gptlget_wallclock__
#define gptlget_wallclock_latest gptlget_wallclock_latest__
#define gptlget_threadwork gptlget_threadwork__
#define gptlstartstop_val gptlstartstop_val__
#define gptlget_eventvalue gptlget_eventvalue__
#define gptlget_nregions gptlget_nregions__
#define gptlget_regionname gptlget_regionname__
#define gptlget_memusage gptlget_memusage__
#define gptlprint_memusage gptlprint_memusage__
#define gptlprint_rusage gptlprint_rusage__
#define gptlnum_errors gptlnum_errors__
#define gptlnum_warn gptlnum_warn__
#define gptlget_count gptlget_count__
#define gptl_papilibraryinit gptl_papilibraryinit__
#define gptlevent_name_to_code gptlevent_name_to_code__
#define gptlevent_code_to_name gptlevent_code_to_name__
#define gptlcompute_chunksize gptlcompute_chunksize__
#define gptlget_gpu_props gptlget_gpu_props__
#define gptlcudadevsync gptlcudadevsync_

#endif

extern "C" {

/* Local function prototypes */
__host__ int gptlinitialize (void);
__host__ int gptlfinalize (void);
__host__ int gptlpr (int *procid);
__host__ int gptlpr_file (char *file, int nc);
__host__ int gptlpr_summary (int *fcomm);
__host__ int gptlpr_summary_file (int *fcomm, char *name, int nc);
__host__ int gptlbarrier (int *fcomm, char *name, int nc);
__host__ int gptlreset (void);
__host__ int gptlreset_timer (char *name, int nc);
__host__ int gptlstamp (double *wall, double *usr, double *sys);
__host__ int gptlstart (char *name, int nc);
__host__ int gptlinit_handle (char *name, int *, int nc);
__host__ int gptlstart_handle (char *name, int *, int nc);
__host__ int gptlstop (char *name, int nc);
__host__ int gptlstop_handle (char *name, int *, int nc);
__host__ int gptlsetoption (int *option, int *val);
__host__ int gptlenable (void);
__host__ int gptldisable (void);
__host__ int gptlsetutr (int *option);
__host__ int gptlquery (const char *name, int *t, int *count, int *onflg, double *wallclock, 
			double *usr, double *sys, long long *papicounters_out, int *maxcounters, 
			int nc);
__host__ int gptlget_wallclock (const char *name, int *t, double *value, int nc);
__host__ int gptlget_wallclock_last (const char *name, int *t, double *value, int nc);
__host__ int gptlget_threadwork (const char *name, double *maxwork, double *imbal, int nc);
__host__ int gptlstartstop_val (const char *name, double *value, int nc);
__host__ int gptlget_eventvalue (const char *timername, const char *eventname, int *t, double *value, 
				 int nc1, int nc2);
__host__ int gptlget_nregions (int *t, int *nregions);
__host__ int gptlget_regionname (int *t, int *region, char *name, int nc);
__host__ int gptlget_memusage (int *size, int *rss, int *share, int *text, int *datastack);
__host__ int gptlprint_memusage (const char *str, int nc);
__host__ int gptlprint_rusage (const char *str, int nc);
__host__ int gptlnum_errors (void);
__host__ int gptlnum_warn (void);
__host__ int gptlget_count (char *, int *, int *, int);
#ifdef HAVE_PAPI
__host__ int gptl_papilibraryinit (void);
__host__ int gptlevent_name_to_code (const char *str, int *code, int nc);
__host__ int gptlevent_code_to_name (int *code, char *str, int nc);
#endif
__host__ int gptlcompute_chunksize (int *, int *);
__host__ int gptlget_gpu_props (int *, int *,int *, int *,int *, int *);
__host__ int gptldevsync (void);

/* Fortran wrapper functions start here */
__host__
int gptlinitialize (void)
{
  return GPTLinitialize ();
}

__host__
int gptlfinalize (void)
{
  return GPTLfinalize ();
}

__host__
int gptlpr (int *procid)
{
  return GPTLpr (*procid);
}

__host__
int gptlpr_file (char *file, int nc)
{
  char locfile[nc+1];
  int ret;

  snprintf (locfile, nc+1, "%s", file);

  ret = GPTLpr_file (locfile);
  return ret;
}

__host__
int gptlpr_summary (int *fcomm)
{
  int ret;

#ifdef HAVE_LIBMPI
  MPI_Comm ccomm;
  ccomm = MPI_Comm_f2c (*fcomm);
  ret = GPTLpr_summary (ccomm);
#else
  ret = GPTLpr_summary (0);
#endif
  return ret;
}

__host__
int gptlpr_summary_file (int *fcomm, char *outfile, int nc)
{
  char locfile[nc+1];
  int ret;

#ifdef HAVE_LIBMPI
  MPI_Comm ccomm;

  snprintf (locfile, nc+1, "%s", outfile);
  ccomm = MPI_Comm_f2c (*fcomm);
  ret = GPTLpr_summary_file (ccomm, locfile);
#else
  snprintf (locfile, nc+1, "%s", outfile);
  ret = GPTLpr_summary_file (0, locfile);
#endif
  return ret;
}

__host__
int gptlbarrier (int *fcomm, char *name, int nc)
{
#ifdef HAVE_LIBMPI
  MPI_Comm ccomm;
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  ccomm = MPI_Comm_f2c (*fcomm);
  return GPTLbarrier (ccomm, cname);
#else
  return GPTLerror ("Either HAVE_LIBMPI not set so cannot call GPTLbarrier\n");
#endif
}

__host__
int gptlreset (void)
{
  return GPTLreset ();
}

__host__
int gptlreset_timer (char *name, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLreset_timer (cname);
}

__host__
int gptlstamp (double *wall, double *usr, double *sys)
{
  return GPTLstamp (wall, usr, sys);
}

__host__
int gptlstart (char *name, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLstart (cname);
}

__host__
int gptlinit_handle (char *name, int *handle, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLinit_handle (cname, handle);
}

__host__
int gptlstart_handle (char *name, int *handle, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLstart_handle (cname, handle);
}

__host__
int gptlstop (char *name, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLstop (cname);
}

__host__
int gptlstop_handle (char *name, int *handle, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLstop_handle (cname, handle);
}

__host__
int gptlsetoption (int *option, int *val)
{
  return GPTLsetoption (*option, *val);
}

__host__
int gptlenable (void)
{
  return GPTLenable ();
}

__host__
int gptldisable (void)
{
  return GPTLdisable ();
}

__host__
int gptlsetutr (int *option)
{
  return GPTLsetutr (*option);
}

__host__
int gptlquery (const char *name, int *t, int *count, int *onflg, double *wallclock, 
	       double *usr, double *sys, long long *papicounters_out, int *maxcounters, 
	       int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLquery (cname, *t, count, onflg, wallclock, usr, sys, papicounters_out, *maxcounters);
}

__host__
int gptlquerycounters (const char *name, int *t, long long *papicounters_out, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLquerycounters (cname, *t, papicounters_out);
}

__host__
int gptlget_wallclock (const char *name, int *t, double *value, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';

  return GPTLget_wallclock (cname, *t, value);
}

__host__
int gptlget_wallclock_latest (const char *name, int *t, double *value, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';

  return GPTLget_wallclock_latest (cname, *t, value);
}

__host__
int gptlget_threadwork (const char *name, double *maxwork, double *imbal, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';

  return GPTLget_threadwork (cname, maxwork, imbal);
}

__host__
int gptlstartstop_val (const char *name, double *value, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLstartstop_val (cname, *value);
}

__host__
int gptlget_eventvalue (const char *timername, const char *eventname, int *t, double *value, 
			int nc1, int nc2)
{
  char ctimername[nc1+1];
  char ceventname[nc2+1];

  strncpy (ctimername, timername, nc1);
  ctimername[nc1] = '\0';

  strncpy (ceventname, eventname, nc2);
  ceventname[nc2] = '\0';

  return GPTLget_eventvalue (ctimername, ceventname, *t, value);
}

__host__
int gptlget_nregions (int *t, int *nregions)
{
  return GPTLget_nregions (*t, nregions);
}

__host__
int gptlget_regionname (int *t, int *region, char *name, int nc)
{
  int n;
  int ret;

  ret = GPTLget_regionname (*t, *region, name, nc);
  /* Turn nulls into spaces for fortran */
  for (n = 0; n < nc; ++n)
    if (name[n] == '\0')
      name[n] = ' ';
  return ret;
}

__host__
int gptlget_memusage (int *size, int *rss, int *share, int *text, int *datastack)
{
  return GPTLget_memusage (size, rss, share, text, datastack);
}

__host__
int gptlprint_memusage (const char *str, int nc)
{
  char cname[nc+1];

  strncpy (cname, str, nc);
  cname[nc] = '\0';
  return GPTLprint_memusage (cname);
}

__host__
int gptlprint_rusage (const char *str, int nc)
{
  char cname[nc+1];

  strncpy (cname, str, nc);
  cname[nc] = '\0';
  return GPTLprint_rusage (cname);
}

__host__
int gptlnum_errors (void)
{
  return GPTLnum_errors ();
}

__host__
int gptlnum_warn (void)
{
  return GPTLnum_warn ();
}

__host__
int gptlget_count (char *name, int *t, int *count, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';

  return GPTLget_count (cname, *t, count);
}

#ifdef HAVE_PAPI
#include <papi.h>

__host__
int gptl_papilibraryinit (void)
{
  return GPTL_PAPIlibraryinit ();;
}

__host__
int gptlevent_name_to_code (const char *str, int *code, int nc)
{
  char cname[PAPI_MAX_STR_LEN+1];
  int numchars = MIN (nc, PAPI_MAX_STR_LEN);

  strncpy (cname, str, numchars);
  cname[numchars] = '\0';

  /* "code" is an int* and is an output variable */
  return GPTLevent_name_to_code (cname, code);
}

__host__
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
    return GPTLerror ("");
  }
  return 0;
}
#endif

__host__
int gptlcompute_chunksize (int *oversub, int *inner_iter_count)
{
  return GPTLcompute_chunksize (*oversub, *inner_iter_count);
}

__host__
int gptlget_gpu_props (int *khz, int *warpsize, int *devnum, int *SMcount, int *cores_per_sm,
		       int *cores_per_gpu)
{
  return GPTLget_gpu_props (khz, warpsize, devnum, SMcount, cores_per_sm, cores_per_gpu);
}

__host__
int gptlcudadevsync (void)
{
  return GPTLcudadevsync ();
}

}
