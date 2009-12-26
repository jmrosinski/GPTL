/*
** $Id: f_wrappers.c,v 1.47 2009-12-26 19:27:22 rosinski Exp $
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
#define gptlget_wallclock GPTLGET_WALLCLOCK
#define gptlget_eventvalue GPTLGET_EVENTVALUE
#define gptlget_nregions GPTLGET_NREGIONS
#define gptlget_regionname GPTLGET_REGIONNAME
#define gptlget_memusage GPTLGET_MEMUSAGE
#define gptlprint_memusage GPTLPRINT_MEMUSAGE
#define gptl_papilibraryinit GPTL_PAPILIBRARYINIT
#define gptlevent_name_to_code GPTLEVENT_NAME_TO_CODE
#define gptlevent_code_to_name GPTLEVENT_CODE_TO_NAME
#define gptlis_initialized GPTLIS_INITIALIZED
#define gptlpr_has_been_called GPTLPR_HAS_BEEN_CALLED

#if ( defined ENABLE_PMPI )
#define mpi_init MPI_INIT
#define mpi_finalize MPI_FINALIZE
#define mpi_send MPI_SEND
#define mpi_recv MPI_RECV
#define mpi_sendrecv MPI_SENDRECV
#define mpi_isend MPI_ISEND
#define mpi_irecv MPI_IRECV
#define mpi_wait MPI_WAIT
#define mpi_waitall MPI_WAITALL
#define mpi_barrier MPI_BARRIER
#define mpi_bcast MPI_BCAST
#define mpi_allreduce MPI_ALLREDUCE
#define mpi_gather MPI_GATHER
#define mpi_scatter MPI_SCATTER
#define mpi_alltoall MPI_ALLTOALL
#define mpi_reduce MPI_REDUCE
#endif

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
#define gptlget_wallclock gptlget_wallclock_
#define gptlget_eventvalue gptlget_eventvalue_
#define gptlget_nregions gptlget_nregions_
#define gptlget_regionname gptlget_regionname_
#define gptlget_memusage gptlget_memusage_
#define gptlprint_memusage gptlprint_memusage_
#define gptl_papilibraryinit gptl_papilibraryinit_
#define gptlevent_name_to_code gptlevent_name_to_code_
#define gptlevent_code_to_name gptlevent_code_to_name_
#define gptlis_initialized gptlis_initialized_
#define gptlpr_has_been_called gptlpr_has_been_called_

#if ( defined ENABLE_PMPI )
#define mpi_init mpi_init_
#define mpi_finalize mpi_finalize_
#define mpi_send mpi_send_
#define mpi_recv mpi_recv_
#define mpi_sendrecv mpi_sendrecv_
#define mpi_isend mpi_isend_
#define mpi_irecv mpi_irecv_
#define mpi_wait mpi_wait_
#define mpi_waitall mpi_waitall_
#define mpi_barrier mpi_barrier_
#define mpi_bcast mpi_bcast_
#define mpi_allreduce mpi_allreduce_
#define mpi_gather mpi_gather_
#define mpi_scatter mpi_scatter_
#define mpi_alltoall mpi_alltoall_
#define mpi_reduce mpi_reduce_
#endif

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
#define gptlget_wallclock gptlget_wallclock__
#define gptlget_eventvalue gptlget_eventvalue__
#define gptlget_nregions gptlget_nregions__
#define gptlget_regionname gptlget_regionname__
#define gptlget_memusage gptlget_memusage__
#define gptlprint_memusage gptlprint_memusage__
#define gptl_papilibraryinit gptl_papilibraryinit__
#define gptlevent_name_to_code gptlevent_name_to_code__
#define gptlevent_code_to_name gptlevent_code_to_name__
#define gptlis_initialized gptlis_initialized__
#define gptlpr_has_been_called gptlpr_has_been_called__

#if ( defined ENABLE_PMPI )
#define mpi_init mpi_init__
#define mpi_finalize mpi_finalize__
#define mpi_send mpi_send__
#define mpi_recv mpi_recv__
#define mpi_sendrecv mpi_sendrecv__
#define mpi_isend mpi_isend__
#define mpi_irecv mpi_irecv__
#define mpi_wait mpi_wait__
#define mpi_waitall mpi_waitall__
#define mpi_barrier mpi_barrier__
#define mpi_bcast mpi_bcast__
#define mpi_allreduce mpi_allreduce__
#define mpi_gather mpi_gather__
#define mpi_scatter mpi_scatter__
#define mpi_alltoall mpi_alltoall__
#define mpi_reduce mpi_reduce__
#endif

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

  if ( ! (locfile = (char *) malloc (nc1+1)))
    return GPTLerror ("gptlpr_file: malloc error\n");

  snprintf (locfile, nc1+1, "%s", file);

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

int gptlbarrier (int *fcomm, char *name, int nc1)
{
  MPI_Comm ccomm;
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (nc1, MAX_CHARS);
  strncpy (cname, name, numchars);
  cname[numchars] = '\0';
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
  return GPTLerror ("gptlpr_summary: Need to build GPTL with #define HAVE_MPI to call this routine\n");
}

int gptlbarrier (void)
{
  return GPTLerror ("gptlbarrier: Need to build GPTL with #define HAVE_MPI to call this routine\n");
}

#endif


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

int gptlget_wallclock (const char *name, int *t, double *value, int nc)
{
  char cname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (nc, MAX_CHARS);
  strncpy (cname, name, numchars);
  cname[numchars] = '\0';

  return GPTLget_wallclock (cname, *t, value);
}

int gptlget_eventvalue (const char *timername, const char *eventname, int *t, double *value, 
			int nc1, int nc2)
{
  char ctimername[MAX_CHARS+1];
  char ceventname[MAX_CHARS+1];
  int numchars;

  numchars = MIN (nc1, MAX_CHARS);
  strncpy (ctimername, timername, numchars);
  ctimername[numchars] = '\0';

  numchars = MIN (nc2, MAX_CHARS);
  strncpy (ceventname, eventname, numchars);
  ceventname[numchars] = '\0';

  return GPTLget_eventvalue (ctimername, ceventname, *t, value);
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
  char cname[PAPI_MAX_STR_LEN+1];
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
    return GPTLerror ("");
  }
  return 0;
}
#endif

int gptlis_initialized ()
{
  return GPTLis_initialized ();
}

int gptlpr_has_been_called ()
{
  return GPTLpr_has_been_called ();
}

#ifdef ENABLE_PMPI
/*
** These routines were adapted from the FPMPI distribution.
** They ensure profiling of Fortran codes, using the routines defined in
** gptl_pmpi.c
*/

#if ( defined FORTRANCAPS )

#define iargc IARGC
#define getarg GETARG

#elif ( defined FORTRANUNDERSCORE )

#define iargc iargc_
#define getarg getarg_

#elif ( defined FORTRANDOUBLEUNDERSCORE )

#define iargc iargc__
#define getarg getarg__

#endif

/*
** mpi_init requires iargc and getarg
*/

#ifdef HAVE_IARGCGETARG
extern int iargc (void);
extern void getarg (int *, char *, int);

void mpi_init (MPI_Fint *ierr)
{
  int Argc;
  int i, argsize = 1024;
  char **Argv, *p;
  int  ArgcSave;           /* Save the argument count */
  char **ArgvSave;         /* Save the pointer to the argument vector */
  char **ArgvValSave;      /* Save entries in the argument vector */

/* Recover the args with the Fortran routines iargc and getarg */
  ArgcSave    = Argc = iargc() + 1; 
  ArgvSave    = Argv = (char **) malloc (Argc * sizeof(char *));
  ArgvValSave = (char**) malloc (Argc * sizeof(char *));
  if ( ! Argv) {
    fprintf (stderr, "Out of space in MPI_INIT");
    *ierr = -1;
    return;
  }

  for (i = 0; i < Argc; i++) {
    ArgvValSave[i] = Argv[i] = (char *) malloc (argsize + 1);
    if ( ! Argv[i]) {
      fprintf (stderr, "Out of space in MPI_INIT");
      *ierr = -1;
      return;
    }
    getarg (&i, Argv[i], argsize);

    /* Trim trailing blanks */
    p = Argv[i] + argsize - 1;
    while (p > Argv[i]) {
      if (*p != ' ') {
	p[1] = '\0';
	break;
      }
      p--;
    }
  }
  
  *ierr = MPI_Init (&Argc, &Argv);
    
  /* Recover space */
  for (i = 0; i < ArgcSave; i++) {
    free (ArgvValSave[i]);
  }
  free (ArgvValSave);
  free (ArgvSave);
}

void mpi_finalize (MPI_Fint *ierr)
{
  *ierr = MPI_Finalize();
}
#endif

#ifndef MPI_STATUS_SIZE
#define MPI_STATUS_SIZE 5
#endif

void mpi_send (void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest,
	       MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *__ierr)
{
  *__ierr = MPI_Send (buf, *count, MPI_Type_f2c (*datatype), *dest, *tag, 
		      MPI_Comm_f2c (*comm));
}

void mpi_recv (void *buf, MPI_Fint *count, MPI_Fint *datatype, 
	       MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, 
	       MPI_Fint *status, MPI_Fint *__ierr)
{
  MPI_Status s;
  /* A local status should be used if MPI_Fint and int are different sizes */
  *__ierr = MPI_Recv (buf, *count, MPI_Type_f2c (*datatype), *source, *tag, 
		      MPI_Comm_f2c (*comm), &s);
  MPI_Status_c2f (&s, status);
}

void mpi_sendrecv (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, 
		   MPI_Fint *dest, MPI_Fint *sendtag, 
		   void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, 
		   MPI_Fint *source, MPI_Fint *recvtag, 
		   MPI_Fint *comm, MPI_Fint *status, MPI_Fint *__ierr)
{
  MPI_Status s;
  *__ierr = MPI_Sendrecv (sendbuf, *sendcount, MPI_Type_f2c (*sendtype),
			  *dest, *sendtag, recvbuf, *recvcount,
			  MPI_Type_f2c (*recvtype), *source, *recvtag,
			  MPI_Comm_f2c (*comm), &s);
  MPI_Status_c2f (&s, status);
}

void mpi_isend (void *buf, MPI_Fint *count, MPI_Fint *datatype, 
		MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, 
		MPI_Fint *request, MPI_Fint *__ierr)
{
  MPI_Request lrequest;
  *__ierr = MPI_Isend (buf, (int) *count, MPI_Type_f2c (*datatype),
		       (int) *dest, (int) *tag, MPI_Comm_f2c (*comm),
		       &lrequest);
  *request = MPI_Request_c2f (lrequest);
}

void mpi_irecv (void *buf, MPI_Fint *count, MPI_Fint *datatype, 
		MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, 
		MPI_Fint *request, MPI_Fint *__ierr)
{
  MPI_Request lrequest;
  *__ierr = MPI_Irecv (buf, (int)*count, MPI_Type_f2c (*datatype),
		       (int)*source,(int)*tag,
		       MPI_Comm_f2c(*comm),&lrequest);
  *request = MPI_Request_c2f (lrequest);
}

void mpi_wait (MPI_Fint *request, MPI_Fint *status, MPI_Fint *__ierr)
{
  MPI_Request lrequest;
  MPI_Status c_status;

  lrequest = MPI_Request_f2c(*request);
  *__ierr = MPI_Wait(&lrequest, &c_status);
  *request = MPI_Request_c2f(lrequest);

  MPI_Status_c2f (&c_status, status);
}

/*
** mpi_waitall was simplified from the FPMPI version.
** This one has a hard limit of LOCAL_ARRAY_SIZE requests.
** If this limit is exceeded, MPI_Abort is called. There is probably
** a better solution.
*/

void mpi_waitall (MPI_Fint *count, MPI_Fint array_of_requests[], 
                  MPI_Fint array_of_statuses[][MPI_STATUS_SIZE], 
                  MPI_Fint *__ierr)
{
  const int LOCAL_ARRAY_SIZE = 1000;
  int i;
  MPI_Request lrequest[LOCAL_ARRAY_SIZE];
  MPI_Status c_status[LOCAL_ARRAY_SIZE];

  if (MPI_STATUS_SIZE != sizeof(MPI_Status)/sizeof(int)) {
    /* Warning - */
    fprintf( stderr, "Warning: The Fortran GPTL code expected the sizeof MPI_Status\n\
 to be %d integers but it is %d.  Rebuild GPTL and make sure that the\n \
 correct value is found and set in f_wrappers.c\n", MPI_STATUS_SIZE,
	     (int) (sizeof(MPI_Status)/sizeof(int)) );
    fprintf (stderr, "Aborting...\n");
    (void) MPI_Abort (MPI_COMM_WORLD, -1);
  }

  if ((int) *count > LOCAL_ARRAY_SIZE) {
    fprintf (stderr, "mpi_waitall: %d is too many requests: recompile f_wrappers.c "
	     "with LOCAL_ARRAY_SIZE > %d\n", (int)*count, LOCAL_ARRAY_SIZE);
    fprintf (stderr, "Aborting...\n");
    (void) MPI_Abort (MPI_COMM_WORLD, -1);
  }

  if ((int)*count > 0) {
    for (i = 0; i < (int)*count; i++) {
      lrequest[i] = MPI_Request_f2c (array_of_requests[i]);
    }

    *__ierr = MPI_Waitall ((int)*count, lrequest, c_status);
    /* By checking for lrequest[i] = 0, we handle persistent requests */
    for (i = 0; i < (int)*count; i++) {
      array_of_requests[i] = MPI_Request_c2f (lrequest[i]);
    }
  } else {
    *__ierr = MPI_Waitall ((int)*count, (MPI_Request *)0, c_status);
  }

  for (i = 0; i < (int)*count; i++) 
    MPI_Status_c2f (&(c_status[i]), &(array_of_statuses[i][0]));
}

void mpi_barrier (MPI_Fint *comm, MPI_Fint *__ierr)
{
  *__ierr = MPI_Barrier (MPI_Comm_f2c (*comm));
}

void mpi_bcast (void *buffer, MPI_Fint *count, MPI_Fint *datatype, 
		MPI_Fint *root, MPI_Fint *comm, MPI_Fint *__ierr)
{
  *__ierr = MPI_Bcast (buffer, *count, MPI_Type_f2c (*datatype), *root, 
		       MPI_Comm_f2c (*comm));
}

void mpi_allreduce (void *sendbuf, void *recvbuf, MPI_Fint *count, 
		    MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
		    MPI_Fint *__ierr)
{
  *__ierr = MPI_Allreduce (sendbuf, recvbuf, *count, MPI_Type_f2c (*datatype),
			   MPI_Op_f2c (*op), MPI_Comm_f2c (*comm));
}

void mpi_gather (void *sendbuf, MPI_Fint *sendcnt, MPI_Fint *sendtype,
		 void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, 
		 MPI_Fint *root, MPI_Fint *comm, MPI_Fint *__ierr)
{
  *__ierr = MPI_Gather (sendbuf, *sendcnt, MPI_Type_f2c (*sendtype),
			recvbuf, *recvcount, MPI_Type_f2c (*recvtype), *root,
			MPI_Comm_f2c (*comm));
}

void mpi_scatter (void *sendbuf, MPI_Fint *sendcnt, MPI_Fint *sendtype, 
		  void *recvbuf, MPI_Fint *recvcnt, MPI_Fint *recvtype, 
		  MPI_Fint *root, MPI_Fint *comm, MPI_Fint *__ierr)
{
  *__ierr = MPI_Scatter (sendbuf, *sendcnt, MPI_Type_f2c (*sendtype), 
			 recvbuf, *recvcnt, MPI_Type_f2c (*recvtype),
			 *root, MPI_Comm_f2c (*comm));
}

void mpi_alltoall (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype,
		   void *recvbuf, MPI_Fint *recvcnt, MPI_Fint *recvtype, 
		   MPI_Fint *comm, MPI_Fint *__ierr)
{
  *__ierr = MPI_Alltoall (sendbuf, *sendcount, MPI_Type_f2c(*sendtype),
			  recvbuf, *recvcnt, MPI_Type_f2c(*recvtype), 
			  MPI_Comm_f2c (*comm));
}

void mpi_reduce (void *sendbuf, void *recvbuf, MPI_Fint *count, 
		 MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, 
		 MPI_Fint *comm, MPI_Fint *__ierr)
{
  *__ierr = MPI_Reduce (sendbuf, recvbuf, *count, MPI_Type_f2c(*datatype),
			MPI_Op_f2c(*op), *root, MPI_Comm_f2c(*comm));
}

#endif
