//#define _GLIBCXX_CMATH
/*
** f_wrappers.c
**
** Author: Jim Rosinski
** 
** Fortran wrappers for timing library routines
*/

#ifdef HAVE_MPI
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
#define gptlget_wallclock gptlget_wallclock_
#define gptlget_wallclock_latest gptlget_wallclock_latest_
#define gptlget_threadwork gptlget_threadwork_
#define gptlstartstop_val gptlstartstop_val_
#define gptlget_nregions gptlget_nregions_
#define gptlget_regionname gptlget_regionname_
#define gptlget_memusage gptlget_memusage_
#define gptlprint_memusage gptlprint_memusage_
#define gptlprint_rusage gptlprint_rusage_
#define gptlnum_errors gptlnum_errors_
#define gptlnum_warn gptlnum_warn_
#define gptlget_count gptlget_count_
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
#define gptlget_wallclock gptlget_wallclock__
#define gptlget_wallclock_latest gptlget_wallclock_latest__
#define gptlget_threadwork gptlget_threadwork__
#define gptlstartstop_val gptlstartstop_val__
#define gptlget_nregions gptlget_nregions__
#define gptlget_regionname gptlget_regionname__
#define gptlget_memusage gptlget_memusage__
#define gptlprint_memusage gptlprint_memusage__
#define gptlprint_rusage gptlprint_rusage__
#define gptlnum_errors gptlnum_errors__
#define gptlnum_warn gptlnum_warn__
#define gptlget_count gptlget_count__
#define gptlcompute_chunksize gptlcompute_chunksize__
#define gptlget_gpu_props gptlget_gpu_props__
#define gptlcudadevsync gptlcudadevsync_

#endif

extern "C" {

/* Local function prototypes */
int gptlinitialize (void);
int gptlfinalize (void);
int gptlpr (int *procid);
int gptlpr_file (char *file, int nc);
#ifdef HAVE_MPI
int gptlpr_summary (int *fcomm);
int gptlpr_summary_file (int *fcomm, char *name, int nc);
int gptlbarrier (int *fcomm, char *name, int nc);
#else
int gptlpr_summary (void);
int gptlpr_summary_file (char *name, int nc);
int gptlbarrier (void);
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
	       double *usr, double *sys, int nc);
int gptlget_wallclock (const char *name, int *t, double *value, int nc);
int gptlget_wallclock_last (const char *name, int *t, double *value, int nc);
int gptlget_threadwork (const char *name, double *maxwork, double *imbal, int nc);
int gptlstartstop_val (const char *name, double *value, int nc);
int gptlget_nregions (int *t, int *nregions);
int gptlget_regionname (int *t, int *region, char *name, int nc);
int gptlget_memusage (int *size, int *rss, int *share, int *text, int *datastack);
int gptlprint_memusage (const char *str, int nc);
int gptlprint_rusage (const char *str, int nc);
int gptlnum_errors (void);
int gptlnum_warn (void);
int gptlget_count (char *, int *, int *, int);
int gptlcompute_chunksize (int *, int *);
int gptlget_gpu_props (int *, int *,int *, int *,int *, int *);
int gptldevsync (void);

/* Fortran wrapper functions start here */
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
  int ret;

  snprintf (locfile, nc+1, "%s", file);

  ret = GPTLpr_file (locfile);
  free (locfile);
  return ret;
}

#ifdef HAVE_MPI

int gptlpr_summary (int *fcomm)
{
  MPI_Comm ccomm;
#ifdef HAVE_COMM_F2C
  ccomm = MPI_Comm_f2c (*fcomm);
#else
  /* Punt and try just casting the Fortran communicator */
  ccomm = (MPI_Comm) *fcomm;
#endif
  return GPTLpr_summary (ccomm);
}

int gptlpr_summary_file (int *fcomm, char *outfile, int nc)
{
  MPI_Comm ccomm;
  char locfile[nc+1];
  int ret;

  snprintf (locfile, nc+1, "%s", outfile);

#ifdef HAVE_COMM_F2C
  ccomm = MPI_Comm_f2c (*fcomm);
#else
  /* Punt and try just casting the Fortran communicator */
  ccomm = (MPI_Comm) *fcomm;
#endif
  ret = GPTLpr_summary_file (ccomm, locfile);
  free (locfile);
  return ret;
}

int gptlbarrier (int *fcomm, char *name, int nc)
{
  MPI_Comm ccomm;
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
#ifdef HAVE_COMM_F2C
  ccomm = MPI_Comm_f2c (*fcomm);
#else
  /* Punt and try just casting the Fortran communicator */
  ccomm = (MPI_Comm) *fcomm;
#endif
  return GPTLbarrier (ccomm, cname);
}

#else

int gptlpr_summary (void)
{
  return GPTLpr_summary ();
}

int gptlpr_summary_file (char *outfile, int nc)
{
  char locfile[nc+1];
  int ret;

  snprintf (locfile, nc+1, "%s", outfile);
  ret = GPTLpr_summary_file (locfile);
  free (locfile);
  return ret;
}

int gptlbarrier (void)
{
  return GPTLerror ("gptlbarrier: Need to build GPTL with #define HAVE_MPI to enable this routine\n");
}

#endif


int gptlreset (void)
{
  return GPTLreset ();
}

int gptlreset_timer (char *name, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLreset_timer (cname);
}

int gptlstamp (double *wall, double *usr, double *sys)
{
  return GPTLstamp (wall, usr, sys);
}

int gptlstart (char *name, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLstart (cname);
}

int gptlinit_handle (char *name, int *handle, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLinit_handle (cname, handle);
}

int gptlstart_handle (char *name, int *handle, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLstart_handle (cname, handle);
}

int gptlstop (char *name, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLstop (cname);
}

int gptlstop_handle (char *name, int *handle, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLstop_handle (cname, handle);
}

int gptlsetoption (int *option, int *val)
{
  return GPTLsetoption (*option, *val);
}

int gptlenable (void)
{
  return GPTLenable ();
}

int gptldisable (void)
{
  return GPTLdisable ();
}

int gptlsetutr (int *option)
{
  return GPTLsetutr (*option);
}

int gptlquery (const char *name, int *t, int *count, int *onflg, double *wallclock, 
	       double *usr, double *sys, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLquery (cname, *t, count, onflg, wallclock, usr, sys);
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

  strncpy (cname, name, nc);
  cname[nc] = '\0';

  return GPTLget_threadwork (cname, maxwork, imbal);
}

int gptlstartstop_val (const char *name, double *value, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';
  return GPTLstartstop_val (cname, *value);
}

int gptlget_nregions (int *t, int *nregions)
{
  return GPTLget_nregions (*t, nregions);
}

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

int gptlget_memusage (int *size, int *rss, int *share, int *text, int *datastack)
{
  return GPTLget_memusage (size, rss, share, text, datastack);
}

int gptlprint_memusage (const char *str, int nc)
{
  char cname[nc+1];

  strncpy (cname, str, nc);
  cname[nc] = '\0';
  return GPTLprint_memusage (cname);
}

int gptlprint_rusage (const char *str, int nc)
{
  char cname[nc+1];

  strncpy (cname, str, nc);
  cname[nc] = '\0';
  return GPTLprint_rusage (cname);
}

int gptlnum_errors (void)
{
  return GPTLnum_errors ();
}

int gptlnum_warn (void)
{
  return GPTLnum_warn ();
}

int gptlget_count (char *name, int *t, int *count, int nc)
{
  char cname[nc+1];

  strncpy (cname, name, nc);
  cname[nc] = '\0';

  return GPTLget_count (cname, *t, count);
}

int gptlcompute_chunksize (int *oversub, int *inner_iter_count)
{
  return GPTLcompute_chunksize (*oversub, *inner_iter_count);
}

int gptlget_gpu_props (int *khz, int *warpsize, int *devnum, int *SMcount, int *cores_per_sm,
		       int *cores_per_gpu)
{
  return GPTLget_gpu_props (khz, warpsize, devnum, SMcount, cores_per_sm, cores_per_gpu);
}

int gptlcudadevsync (void)
{
  return GPTLcudadevsync ();
}

}
