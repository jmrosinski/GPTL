/*
** f_wrappers.c
**
** Author: Jim Rosinski
** 
** Fortran wrappers for timing library routines
*/

#include "config.h" // Must be first include
#include "private.h" // MAX_CHARS, bool
#include "gptl.h"

#ifdef HAVE_LIBMPI
#include "gptlmpi.h"
#include <mpi.h>
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

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
#define gptlget_wallclock gptlget_wallclock_
#define gptlget_wallclock_latest gptlget_wallclock_latest_
#define gptlget_threadwork gptlget_threadwork_
#define gptlstartstop_val gptlstartstop_val_
#define gptlget_eventvalue gptlget_eventvalue_
#define gptlget_nregions gptlget_nregions_
#define gptlget_regionname gptlget_regionname_
#define gptlget_memusage gptlget_memusage_
#define gptlprint_memusage gptlprint_memusage_
#define gptlget_procsiz gptlget_procsiz_
#define gptlnum_errors gptlnum_errors_
#define gptlnum_warn gptlnum_warn_
#define gptlget_count gptlget_count_
#define gptl_papilibraryinit gptl_papilibraryinit_
#define gptlevent_name_to_code gptlevent_name_to_code_
#define gptlevent_code_to_name gptlevent_code_to_name_
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
#define gptlget_wallclock gptlget_wallclock__
#define gptlget_wallclock_latest gptlget_wallclock_latest__
#define gptlget_threadwork gptlget_threadwork__
#define gptlstartstop_val gptlstartstop_val__
#define gptlget_eventvalue gptlget_eventvalue__
#define gptlget_nregions gptlget_nregions__
#define gptlget_regionname gptlget_regionname__
#define gptlget_memusage gptlget_memusage__
#define gptlprint_memusage gptlprint_memusage__
#define gptlget_procsiz gptlget_procsiz__
#define gptlnum_errors gptlnum_errors__
#define gptlnum_warn gptlnum_warn__
#define gptlget_count gptlget_count__
#define gptl_papilibraryinit gptl_papilibraryinit__
#define gptlevent_name_to_code gptlevent_name_to_code__
#define gptlevent_code_to_name gptlevent_code_to_name__
#define gptlget_gpu_props gptlget_gpu_props__
#define gptlcudadevsync gptlcudadevsync_

#endif

// Local function prototypes: Everything callable by Fortran requires C linkage
#ifdef __cplusplus
extern "C" {
#endif
int gptlinitialize (void);
int gptlfinalize (void);
int gptlpr (int *procid);
int gptlpr_file (char *file, int nc);
#ifdef HAVE_LIBMPI
int gptlpr_summary (int *fcomm);
int gptlpr_summary_file (int *fcomm, char *name, int nc);
int gptlbarrier (int *fcomm, char *name, int nc);
#endif
int gptlreset (void);
int gptlreset_timer (char *name, int nc);
int gptlstamp (double *wall, double *usr, double *sys);
int gptlstart (char *name, int nc);
int gptlinit_handle (char *name, int *, int nc);
int gptlstart_handle (char *name, int *, int nc);
int gptlstop (char *name, int nc);
int gptlstop_handle (char *name, int *, int nc);
int gptlsetoption (int *option, int *val);
int gptlenable (void);
int gptldisable (void);
int gptlsetutr (int *option);
int gptlquery (const char *name, int *t, int *count, int *onflg, double *wallclock, 
	       double *usr, double *sys, long long *papicounters_out, int *maxcounters, 
	       int nc);
int gptlget_wallclock (const char *name, int *t, double *value, int nc);
int gptlget_wallclock_last (const char *name, int *t, double *value, int nc);
int gptlget_threadwork (const char *name, double *maxwork, double *imbal, int nc);
int gptlstartstop_val (const char *name, double *value, int nc);
int gptlget_eventvalue (const char *timername, const char *eventname, int *t, double *value, 
			int nc1, int nc2);
int gptlget_nregions (int *t, int *nregions);
int gptlget_regionname (int *t, int *region, char *name, int nc);
int gptlget_memusage (float *);
int gptlprint_memusage (const char *str, int nc);
int gptlget_procsiz (float *, float *);
int gptlnum_errors (void);
int gptlnum_warn (void);
int gptlget_count (char *, int *, int *, int);
#ifdef HAVE_PAPI
int gptl_papilibraryinit (void);
int gptlevent_name_to_code (const char *str, int *code, int nc);
int gptlevent_code_to_name (int *code, char *str, int nc);
#endif
#ifdef ENABLE_CUDA
int gptlget_gpu_props (int *, int *,int *, int *,int *, int *);
int gptlcudadevsync (void);
#endif

// Fortran wrapper functions start here
int gptlinitialize (void)
{
  return GPTLinitialize ();
}

int gptlfinalize (void)
{
  return GPTLfinalize ();
}

int gptlpr (int *procid)
{
  return GPTLpr (*procid);
}

int gptlpr_file (char *file, int nc)
{
  char locfile[nc+1];
  snprintf (locfile, nc+1, "%s", file);
  return GPTLpr_file (locfile);
}

#ifdef HAVE_LIBMPI
int gptlpr_summary (int *fcomm)
{
  MPI_Comm ccomm;
  ccomm = MPI_Comm_f2c (*fcomm);
  return GPTLpr_summary (ccomm);
}

int gptlpr_summary_file (int *fcomm, char *outfile, int nc)
{
  MPI_Comm ccomm;
  char locfile[nc+1];
  snprintf (locfile, nc+1, "%s", outfile);
  ccomm = MPI_Comm_f2c (*fcomm);
  return GPTLpr_summary_file (ccomm, locfile);
}

int gptlbarrier (int *fcomm, char *name, int nc)
{
  MPI_Comm ccomm;
  char cname[nc+1];
  ccomm = MPI_Comm_f2c (*fcomm);
  if (name[nc-1] == '\0') {
    return GPTLbarrier (ccomm, name);
  } else {
    strncpy (cname, name, nc);
    cname[nc] = '\0';
    return GPTLbarrier (ccomm, cname);
  }
}
#endif

int gptlreset (void) {return GPTLreset ();}
int gptlreset_timer (char *name, int nc)
{
  char cname[nc+1];
  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLreset_timer (cname);
}

int gptlstamp (double *wall, double *usr, double *sys) {return GPTLstamp (wall, usr, sys);}
int gptlstart (char *name, int nc)
{
  char cname[nc+1];
  // Check for name already null-terminated for efficiency
  if (name[nc-1] == '\0') {
    return GPTLstart (name);
  } else {
    strncpy (cname, name, nc);
    cname[nc] = '\0';
    return GPTLstart (cname);
  }
}

int gptlinit_handle (char *name, int *handle, int nc)
{
  char cname[nc+1];
  // Check for name already null-terminated for efficiency
  if (name[nc-1] == '\0') {
    return GPTLinit_handle (name, handle);
  } else {
    strncpy (cname, name, nc);
    cname[nc] = '\0';
    return GPTLinit_handle (cname, handle);
  }
}

int gptlstart_handle (char *name, int *handle, int nc)
{
  char cname[nc+1];
  // Check for name already null-terminated for efficiency
  if (name[nc-1] == '\0') {
    return GPTLstart_handle (name, handle);
  } else {
    strncpy (cname, name, nc);
    cname[nc] = '\0';
    return GPTLstart_handle (cname, handle);
  }
}

int gptlstop (char *name, int nc)
{
  char cname[nc+1];
  // Check for name already null-terminated for efficiency
  if (name[nc-1] == '\0') {
    return GPTLstop (name);
  } else {
    strncpy (cname, name, nc);
    cname[nc] = '\0';
    return GPTLstop (cname);
  }
}

int gptlstop_handle (char *name, int *handle, int nc)
{
  char cname[nc+1];
  // Check for name already null-terminated for efficiency
  if (name[nc-1] == '\0') {
    return GPTLstop_handle (name, handle);
  } else {
    strncpy (cname, name, nc);
    cname[nc] = '\0';
    return GPTLstop_handle (cname, handle);
  }
}

int gptlsetoption (int *option, int *val) {return GPTLsetoption (*option, *val);}
int gptlenable (void) {return GPTLenable ();}
int gptldisable (void) {return GPTLdisable ();}
int gptlsetutr (int *option) {return GPTLsetutr (*option);}
int gptlquery (const char *name, int *t, int *count, int *onflg, double *wallclock, 
	       double *usr, double *sys, long long *papicounters_out, int *maxcounters, 
	       int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLquery (cname, *t, count, onflg, wallclock, usr, sys, papicounters_out, *maxcounters);
}

int gptlget_wallclock (const char *name, int *t, double *value, int nc)
{
  char cname[nc+1];
  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLget_wallclock (cname, *t, value);
}

int gptlget_wallclock_latest (const char *name, int *t, double *value, int nc)
{
  char cname[nc+1];
  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLget_wallclock_latest (cname, *t, value);
}

int gptlget_threadwork (const char *name, double *maxwork, double *imbal, int nc)
{
  char cname[nc+1];
  // Check for name already null-terminated for efficiency
  if (name[nc-1] == '\0') {
    return GPTLget_threadwork (name, maxwork, imbal);
  } else {
    strncpy (cname, name, nc);
    cname[nc] = '\0';
    return GPTLget_threadwork (cname, maxwork, imbal);
  }
}

int gptlstartstop_val (const char *name, double *value, int nc)
{
  char cname[nc+1];
  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLstartstop_val (cname, *value);
}

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

int gptlget_nregions (int *t, int *nregions) {return GPTLget_nregions (*t, nregions);}
int gptlget_regionname (int *t, int *region, char *name, int nc)
{
  int n;
  int ret;
  ret = GPTLget_regionname (*t, *region, name, nc);
  // Turn nulls into spaces for fortran
  for (n = 0; n < nc; ++n)
    if (name[n] == '\0')
      name[n] = ' ';
  return ret;
}

int gptlget_memusage (float *rss) {return GPTLget_memusage (rss);}
int gptlprint_memusage (const char *str, int nc)
{
  char cname[nc+1];
  strncpy (cname, str, nc);
  cname[nc] = '\0';
  return GPTLprint_memusage (cname);
}

int gptlget_procsiz (float *procsiz, float *rss) {return GPTLget_procsiz (procsiz, rss);}
int gptlnum_errors (void) {return GPTLnum_errors ();}
int gptlnum_warn (void) {return GPTLnum_warn ();}
int gptlget_count (char *name, int *t, int *count, int nc)
{
  char cname[nc+1];
  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLget_count (cname, *t, count);
}

#ifdef HAVE_PAPI
#include <papi.h>

int gptl_papilibraryinit (void) {return GPTL_PAPIlibraryinit ();}
int gptlevent_name_to_code (const char *str, int *code, int nc)
{
  char cname[PAPI_MAX_STR_LEN+1];
  int numchars = MIN (nc, PAPI_MAX_STR_LEN);
  strncpy (cname, str, numchars);
  cname[numchars] = '\0';
  // "code" is an int* and is an output variable
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
    return GPTLerror ("");
  }
  return 0;
}
#endif

#ifdef ENABLE_CUDA
int gptlget_gpu_props (int *khz, int *warpsize, int *devnum, int *SMcount, int *cores_per_sm,
		       int *cores_per_gpu)
{
  return GPTLget_gpu_props (khz, warpsize, devnum, SMcount, cores_per_sm, cores_per_gpu);
}

int gptlcudadevsync (void)
{
  return GPTLcudadevsync ();
}
#endif

#ifdef __cplusplus
}
#endif
