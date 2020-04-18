/*
** $Id: gptl_papi.c,v 1.79 2011-03-28 20:55:19 rosinski Exp $
**
** Author: Jim Rosinski
**
** Contains routines which interface to PAPI library
*/
#include "config.h" /* Must be first include. */
 
#include "private.h"
#include "gptl.h"

#include <papi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if ( defined THREADED_OMP )
#include <omp.h>
#elif ( defined THREADED_PTHREADS )
#include <pthread.h>
#endif

static int papieventlist[MAX_AUX];        /* list of PAPI events to be counted */
static Pr_event pr_event[MAX_AUX];        /* list of events (PAPI or derived) */

/* Derived events */
static const Entry derivedtable [] = {
  {GPTL_IPC,    "GPTL_IPC",     "IPC     "},
  {GPTL_LSTPI,  "GPTL_LSTPI",   "LST_frac"},
  {GPTL_DCMRT,  "GPTL_DCMRT",   "DCMISRAT"},
  {GPTL_LSTPDCM,"GPTL_LSTPDCM", "LSTPDCM "},
  {GPTL_L2MRT,  "GPTL_L2MRT",   "L2MISRAT"},
  {GPTL_LSTPL2M,"GPTL_LSTPL2M", "LSTPL2M "},
  {GPTL_L3MRT,  "GPTL_L3MRT",   "L3MISRAT"}
};
static const int nderivedentries = sizeof (derivedtable) / sizeof (Entry);

static int npapievents = 0;              /* number of PAPI events: initialize to 0 */ 
static int nevents = 0;                  /* number of events: initialize to 0 */ 
static int *EventSet;                    /* list of events to be counted by PAPI */
static long_long **papicounters;         /* counters returned from PAPI */

static const int BADCOUNT = -999999;     /* Set counters to this when they are bad */
static bool is_multiplexed = false;      /* whether multiplexed (always start false)*/
static bool persec = true;               /* print PAPI stats per second */
static bool enable_multiplexing = false; /* whether to try multiplexing */
static bool verbose = false;             /* output verbosity */

// Function prototypes: Give C linkage to all
#ifdef __cplusplus
extern "C" {
#endif
  
static int canenable (int);
static int canenable2 (int, int);
static int papievent_is_enabled (int);
static int already_enabled (int);
static int enable (int);
static int getderivedidx (int);

/*
** GPTL_PAPIsetoption: enable or disable PAPI event defined by "counter". Called 
**   from GPTLsetoption.  Since all events are off by default, val=false degenerates
**   to a no-op.  Coded this way to be consistent with the rest of GPTL
**
** Input args: 
**   counter: PAPI counter
**   val:     true or false for enable or disable
**
** Return value: 0 (success) or GPTLerror (failure)
*/

int GPTL_PAPIsetoption (const int counter, const int val)
{
  int n;       // loop index
  int ret;     // return code
  int numidx;  // numerator index
  int idx;     // derived counter index
  char eventname[PAPI_MAX_STR_LEN]; // returned from PAPI_event_code_to_name
  static const char *thisfunc = "GPTL_PAPIsetoption";

  // First check for option which is not an actual counter
  switch (counter) {
  case GPTLverbose:
    // don't printf here--that'd duplicate what's in gptl.cc
    verbose = (bool) val;
    return 0;
  case GPTLmultiplex:
    enable_multiplexing = (bool) val;
    if (verbose)
      printf ("%s: boolean enable_multiplexing = %d\n", thisfunc, val);
    return 0;
  case GPTLpersec:
    persec = (bool) val;
    if (verbose)
      printf ("%s: boolean persec = %d\n", thisfunc, val);
    return 0;
  default:
    break;
  }

  /* 
  ** If val is false, return an error if the event has already been enabled.
  ** Otherwise just warn that attempting to disable a PAPI-based event
  ** that has already been enabled doesn't work--for now it's just a no-op
  */
  if (! val) {
    if (already_enabled (counter))
      return GPTLerror ("%s: already enabled counter %d cannot be disabled\n", thisfunc, counter);
    else
      if (verbose)
	printf ("%s: 'disable' %d currently is just a no-op\n", thisfunc, counter);
    return 0;
  }

  // If the event has already been enabled for printing, exit
  if (already_enabled (counter))
    return GPTLerror ("%s: counter %d has already been enabled\n", thisfunc, counter);

  // Initialize PAPI if it hasn't already been done.
  // From here on down we can assume the intent is to enable (not disable) an option
  if (GPTL_PAPIlibraryinit () < 0)
    return GPTLerror ("%s: PAPI library init error\n", thisfunc);

  // Ensure max nevents won't be exceeded
  if (nevents+1 > MAX_AUX)
    return GPTLerror ("%s: %d is too many events. Value defined in private.h\n",
		      thisfunc, nevents+1);

  // Check derived events
  switch (counter) {
  case GPTL_IPC:
    if ( ! canenable2 (PAPI_TOT_INS, PAPI_TOT_CYC))
      return GPTLerror ("%s: GPTL_IPC unavailable\n", thisfunc);

    idx = getderivedidx (GPTL_IPC);
    pr_event[nevents].event    = derivedtable[idx];
    pr_event[nevents].numidx   = enable (PAPI_TOT_INS);
    pr_event[nevents].denomidx = enable (PAPI_TOT_CYC);
    if (verbose)
      printf ("%s: enabling derived event %s = PAPI_TOT_INS / PAPI_TOT_CYC\n", 
	      thisfunc, pr_event[nevents].event.namestr);
    ++nevents;
    return 0;
  case GPTL_LSTPI:
    idx = getderivedidx (GPTL_LSTPI);
    if (canenable2 (PAPI_LST_INS, PAPI_TOT_INS)) {
      pr_event[nevents].event    = derivedtable[idx];
      pr_event[nevents].numidx   = enable (PAPI_LST_INS);
      pr_event[nevents].denomidx = enable (PAPI_TOT_INS);
      if (verbose)
	printf ("%s: enabling derived event %s = PAPI_LST_INS / PAPI_TOT_INS\n", 
		thisfunc, pr_event[nevents].event.namestr);
    } else if (canenable2 (PAPI_L1_DCA, PAPI_TOT_INS)) {
      pr_event[nevents].event    = derivedtable[idx];
      pr_event[nevents].numidx   = enable (PAPI_L1_DCA);
      pr_event[nevents].denomidx = enable (PAPI_TOT_INS);
      if (verbose)
	printf ("%s: enabling derived event %s = PAPI_L1_DCA / PAPI_TOT_INS\n", 
		thisfunc, pr_event[nevents].event.namestr);
    } else {
      return GPTLerror ("%s: GPTL_LSTPI unavailable\n", thisfunc);
    }
    ++nevents;
    return 0;
  case GPTL_DCMRT:
    if ( ! canenable2 (PAPI_L1_DCM, PAPI_L1_DCA))
      return GPTLerror ("%s: GPTL_DCMRT unavailable\n", thisfunc);

    idx = getderivedidx (GPTL_DCMRT);
    pr_event[nevents].event    = derivedtable[idx];
    pr_event[nevents].numidx   = enable (PAPI_L1_DCM);
    pr_event[nevents].denomidx = enable (PAPI_L1_DCA);
    if (verbose)
      printf ("%s: enabling derived event %s = PAPI_L1_DCM / PAPI_L1_DCA\n", 
	      thisfunc, pr_event[nevents].event.namestr);
    ++nevents;
    return 0;
  case GPTL_LSTPDCM:
    idx = getderivedidx (GPTL_LSTPDCM);
    if (canenable2 (PAPI_LST_INS, PAPI_L1_DCM)) {
      pr_event[nevents].event    = derivedtable[idx];
      pr_event[nevents].numidx   = enable (PAPI_LST_INS);
      pr_event[nevents].denomidx = enable (PAPI_L1_DCM);
      if (verbose)
	printf ("%s: enabling derived event %s = PAPI_LST_INS / PAPI_L1_DCM\n", 
		thisfunc, pr_event[nevents].event.namestr);
    } else if (canenable2 (PAPI_L1_DCA, PAPI_L1_DCM)) {
      pr_event[nevents].event    = derivedtable[idx];
      pr_event[nevents].numidx   = enable (PAPI_L1_DCA);
      pr_event[nevents].denomidx = enable (PAPI_L1_DCM);
      if (verbose)
	printf ("%s: enabling derived event %s = PAPI_L1_DCA / PAPI_L1_DCM\n", 
		thisfunc, pr_event[nevents].event.namestr);
    } else {
      return GPTLerror ("%s: GPTL_LSTPDCM unavailable\n", thisfunc);
    }
    ++nevents;
    return 0;
    // For L2 counts, use TC* instead of DC* to avoid PAPI derived events
  case GPTL_L2MRT:
    if ( ! canenable2 (PAPI_L2_TCM, PAPI_L2_TCA))
      return GPTLerror ("%s: GPTL_L2MRT unavailable\n", thisfunc);

    idx = getderivedidx (GPTL_L2MRT);
    pr_event[nevents].event    = derivedtable[idx];
    pr_event[nevents].numidx   = enable (PAPI_L2_TCM);
    pr_event[nevents].denomidx = enable (PAPI_L2_TCA);
    if (verbose)
      printf ("%s: enabling derived event %s = PAPI_L2_TCM / PAPI_L2_TCA\n", 
	      thisfunc, pr_event[nevents].event.namestr);
    ++nevents;
    return 0;
  case GPTL_LSTPL2M:
    idx = getderivedidx (GPTL_LSTPL2M);
    if (canenable2 (PAPI_LST_INS, PAPI_L2_TCM)) {
      pr_event[nevents].event    = derivedtable[idx];
      pr_event[nevents].numidx   = enable (PAPI_LST_INS);
      pr_event[nevents].denomidx = enable (PAPI_L2_TCM);
      if (verbose)
	printf ("%s: enabling derived event %s = PAPI_LST_INS / PAPI_L2_TCM\n", 
		thisfunc, pr_event[nevents].event.namestr);
    } else if (canenable2 (PAPI_L1_DCA, PAPI_L2_TCM)) {
      pr_event[nevents].event    = derivedtable[idx];
      pr_event[nevents].numidx   = enable (PAPI_L1_DCA);
      pr_event[nevents].denomidx = enable (PAPI_L2_TCM);
      if (verbose)
	printf ("%s: enabling derived event %s = PAPI_L1_DCA / PAPI_L2_TCM\n", 
		thisfunc, pr_event[nevents].event.namestr);
    } else {
      return GPTLerror ("%s: GPTL_LSTPL2M unavailable\n", thisfunc);
    }
    ++nevents;
    return 0;
  case GPTL_L3MRT:
    if ( ! canenable2 (PAPI_L3_TCM, PAPI_L3_TCR))
      return GPTLerror ("%s: GPTL_L3MRT unavailable\n", thisfunc);

    idx = getderivedidx (GPTL_L3MRT);
    pr_event[nevents].event    = derivedtable[idx];
    pr_event[nevents].numidx   = enable (PAPI_L3_TCM);
    pr_event[nevents].denomidx = enable (PAPI_L3_TCR);
    if (verbose)
      printf ("%s: enabling derived event %s = PAPI_L3_TCM / PAPI_L3_TCR\n", 
	      thisfunc, pr_event[nevents].event.namestr);
    ++nevents;
    return 0;
  default:
    break;
  }

  // Check PAPI events: If PAPI_event_code_to_name fails, give up
  if ((ret = PAPI_event_code_to_name (counter, eventname)) != PAPI_OK)
    return GPTLerror ("%s: name not found for counter %d: PAPI_strerror: %s\n", 
		      thisfunc, counter, PAPI_strerror (ret));

  // Truncate eventname, except strip off PAPI_ for the shortest
  numidx = papievent_is_enabled (counter);
  if (numidx >= 0 || canenable (counter)) {
    int nchars;
    pr_event[nevents].event.counter = counter;

    strncpy (pr_event[nevents].event.namestr, eventname, 12);
    pr_event[nevents].event.namestr[12] = '\0';

    nchars = MIN (strlen (&eventname[5]), 8);
    strncpy (pr_event[nevents].event.str8, &eventname[5], nchars);
    pr_event[nevents].event.str8[nchars] = '\0';

    if (numidx >= 0) {
      pr_event[nevents].numidx = numidx;
      pr_event[nevents].denomidx = -1;     // flag says not derived (no denominator)
    } else {   // canenable (counter) is true
      pr_event[nevents].numidx = enable (counter);
      pr_event[nevents].denomidx = -1;     // flag says not derived (no denominator)
    }
  } else {
    return GPTLerror ("%s: Can't enable event %s\n", thisfunc, eventname);
  }

  if (verbose)
    printf ("%s: enabling native event %s\n", thisfunc, pr_event[nevents].event.namestr);

  ++nevents;
  return 0;
}

/*
** canenable: determine whether a PAPI counter can be enabled
**
** Input args: 
**   counter: PAPI counter
**
** Return value: 0 (success) or non-zero (failure)
*/
int canenable (int counter)
{
  char eventname[PAPI_MAX_STR_LEN]; /* returned from PAPI_event_code_to_name */

  if (npapievents+1 > MAX_AUX)
    return false;

  if (PAPI_query_event (counter) != PAPI_OK) {
    (void) PAPI_event_code_to_name (counter, eventname);
    fprintf (stderr, "GPTL: canenable: event %s not available on this arch\n", eventname);
    return false;
  }

  return true;
}

/*
** canenable2: determine whether 2 PAPI counters can be enabled
**
** Input args: 
**   counter1: PAPI counter
**   counter2: PAPI counter
**
** Return value: 0 (success) or non-zero (failure)
*/
int canenable2 (int counter1, int counter2)
{
  char eventname[PAPI_MAX_STR_LEN]; /* returned from PAPI_event_code_to_name */

  if (npapievents+2 > MAX_AUX)
    return false;

  if (PAPI_query_event (counter1) != PAPI_OK) {
    (void) PAPI_event_code_to_name (counter1, eventname);
    return false;
  }

  if (PAPI_query_event (counter2) != PAPI_OK) {
    (void) PAPI_event_code_to_name (counter2, eventname);
    return false;
  }

  return true;
}

/*
** papievent_is_enabled: determine whether a PAPI counter has already been
**   enabled. Used internally to keep track of PAPI counters enabled. A given
**   PAPI counter may occur in the computation of multiple derived events, as
**   well as output directly. E.g. PAPI_SP_OPS is used to compute
**   computational intensity, and floating point ops per instruction.
**
** Input args: 
**   counter: PAPI counter
**
** Return value: index into papieventlist (success) or negative (not found)
*/
int papievent_is_enabled (int counter)
{
  int n;

  for (n = 0; n < npapievents; ++n)
    if (papieventlist[n] == counter)
      return n;
  return -1;
}

/*
** already_enabled: determine whether a PAPI-based event has already been
**   enabled for printing. 
**
** Input args: 
**   counter: PAPI or derived counter
**
** Return value: 1 (true) or 0 (false)
*/
int already_enabled (int counter)
{
  int n;

  for (n = 0; n < nevents; ++n)
    if (pr_event[n].event.counter == counter)
      return 1;
  return 0;
}

/*
** enable: enable a PAPI event. ASSUMES that canenable() has already determined
**   that the event can be enabled.
**
** Input args: 
**   counter: PAPI counter
**
** Return value: index into papieventlist
*/
int enable (int counter)
{
  int n;

  /* If the event is already enabled, return its index */
  for (n = 0; n < npapievents; ++n) {
    if (papieventlist[n] == counter) {
#ifdef DEBUG
      printf ("GPTL: enable: PAPI event %d is %d\n", n, counter);
#endif
      return n;
    }
  }

  /* New event */
  papieventlist[npapievents++] = counter;
  return npapievents-1;
}

/*
** getderivedidx: find the table index of a derived counter
**
** Input args: 
**   counter: derived counter
**
** Return value: index into derivedtable (success) or GPTLerror (failure)
*/
int getderivedidx (int dcounter)
{
  int n;

  for (n = 0; n < nderivedentries; ++n) {
    if (derivedtable[n].counter == dcounter)
      return n;
  }
  return GPTLerror ("GPTL: getderivedidx: failed to find derived counter %d\n", dcounter);
}

/*
** GPTL_PAPIlibraryinit: Call PAPI_library_init if necessary
**
** Return value: 0 (success) or GPTLerror (failure)
*/
int GPTL_PAPIlibraryinit ()
{
  int ret;
  static const char *thisfunc = "GPTL_PAPIlibraryinit";

  if ((ret = PAPI_is_initialized ()) == PAPI_NOT_INITED) {
    if ((ret = PAPI_library_init (PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
      fprintf (stderr, "%s: ret=%d PAPI_VER_CURRENT=%d\n", thisfunc, ret, (int) PAPI_VER_CURRENT);
      return GPTLerror ("%s: PAPI_library_init failure:%s\n", thisfunc, PAPI_strerror (ret));
    }
  }
  return 0;
}

/*
** GPTL_PAPIinitialize(): Initialize the PAPI interface. Called from GPTLinitialize.
**   PAPI_library_init must be called before any other PAPI routines.  
**   PAPI_thread_init is called subsequently if threading is enabled.
**   Finally, allocate space for PAPI counters and start them.
**
** Input args: 
**   maxthreads: number of threads
**
** Return value: 0 (success) or GPTLerror or -1 (failure)
*/
int GPTL_PAPIinitialize (const int maxthreads, const bool verbose_flag, int *nevents_out,
			 Entry *pr_event_out)
{
  int ret;
  int n;
  int t;
  static const char *thisfunc = "GPTL_PAPIinitialize";

  verbose = verbose_flag;

  if (maxthreads < 1)
    return GPTLerror ("%s: maxthreads = %d\n", thisfunc, maxthreads);

  // Ensure that PAPI_library_init has already been called
  if ((ret = GPTL_PAPIlibraryinit ()) < 0)
    return GPTLerror ("%s: GPTL_PAPIlibraryinit failure\n", thisfunc);

  // PAPI_thread_init needs to be called if threading enabled
#if ( defined THREADED_OMP )
  if (PAPI_thread_init ((unsigned long (*)(void)) (omp_get_thread_num)) != PAPI_OK)
    return GPTLerror ("%s: PAPI_thread_init failure\n", thisfunc);
#elif ( defined THREADED_PTHREADS )
  if (PAPI_thread_init ((unsigned long (*)(void)) (pthread_self)) != PAPI_OK)
    return GPTLerror ("%s: PAPI_thread_init failure\n", thisfunc);
#endif

  // allocate and initialize static local space
  EventSet     = (int *)        GPTLallocate (maxthreads * sizeof (int), thisfunc);
  papicounters = (long_long **) GPTLallocate (maxthreads * sizeof (long_long *), thisfunc);

  for (t = 0; t < maxthreads; t++) {
    EventSet[t] = PAPI_NULL;
    papicounters[t] = (long_long *) GPTLallocate (MAX_AUX * sizeof (long_long), thisfunc);
  }

  *nevents_out = nevents;
  for (n = 0; n < nevents; ++n) {
    pr_event_out[n].counter = pr_event[n].event.counter;
    strcpy (pr_event_out[n].namestr, pr_event[n].event.namestr);
    strcpy (pr_event_out[n].str8   , pr_event[n].event.str8);
  }
  return 0;
}

/*
** GPTLcreate_and_start_events: Create and start the PAPI eventset.
**   Threaded routine to create the "event set" (PAPI terminology) and start
**   the counters. This is only done once, and is called from get_thread_num
**   for the first time for the thread.
** 
** Input args: 
**   t: thread number
**
** Return value: 0 (success) or GPTLerror (failure)
*/
int GPTLcreate_and_start_events (const int t)  /* thread number */
{
  int ret;
  int n;
  char eventname[PAPI_MAX_STR_LEN]; // returned from PAPI_event_code_to_name
  static const char *thisfunc = "GPTLcreate_and_start_events";

  // Set the domain to count all contexts. Only needs to be set once for all threads
  if ((ret = PAPI_set_domain (PAPI_DOM_ALL)) != PAPI_OK)
    return GPTLerror ("%s: thread %d failure setting PAPI domain: %s\n", 
		      thisfunc, t, PAPI_strerror (ret));
  
  // Create the event set
  if ((ret = PAPI_create_eventset (&EventSet[t])) != PAPI_OK)
    return GPTLerror ("%s: thread %d failure creating eventset: %s\n", 
		      thisfunc, t, PAPI_strerror (ret));

  if (verbose)
    printf ("%s: successfully created eventset for thread %d\n", thisfunc, t);

  // Add requested events to the event set
  for (n = 0; n < npapievents; n++) {
    if ((ret = PAPI_add_event (EventSet[t], papieventlist[n])) != PAPI_OK) {
      if (verbose) {
	fprintf (stderr, "%s\n", PAPI_strerror (ret));
	ret = PAPI_event_code_to_name (papieventlist[n], eventname);
	fprintf (stderr, "%s: failure adding event:%s\n", thisfunc, eventname);
      }

      if (enable_multiplexing) {
        if (verbose)
	  printf ("Trying multiplexing...\n");
	is_multiplexed = true;
	break;
      } else
	return GPTLerror ("enable_multiplexing is false: giving up\n");
    }
  }

  if (is_multiplexed) {

    // Cleanup the eventset for multiplexing
    if ((ret = PAPI_cleanup_eventset (EventSet[t])) != PAPI_OK)
      return GPTLerror ("%s: %s\n", thisfunc, PAPI_strerror (ret));
    
    if ((ret = PAPI_destroy_eventset (&EventSet[t])) != PAPI_OK)
      return GPTLerror ("%s: %s\n", thisfunc, PAPI_strerror (ret));

    if ((ret = PAPI_create_eventset (&EventSet[t])) != PAPI_OK)
      return GPTLerror ("%s: failure creating eventset: %s\n", thisfunc, PAPI_strerror (ret));
			
    // Assign EventSet to component 0 (cpu). This step is MANDATORY in recent PAPI releases
    // in order to enable event multiplexing
    if ((ret = PAPI_assign_eventset_component (EventSet[t], 0)) != PAPI_OK)
      return GPTLerror ("%s: thread %d failure in PAPI_assign_eventset_component: %s\n", 
			thisfunc, t, PAPI_strerror (ret));

    if ((ret = PAPI_multiplex_init ()) != PAPI_OK)
      return GPTLerror ("%s: failure from PAPI_multiplex_init%s\n", thisfunc, PAPI_strerror (ret));

    if ((ret = PAPI_set_multiplex (EventSet[t])) != PAPI_OK)
      return GPTLerror ("%s: failure from PAPI_set_multiplex: %s\n", thisfunc, PAPI_strerror (ret));

    for (n = 0; n < npapievents; n++) {
      if ((ret = PAPI_add_event (EventSet[t], papieventlist[n])) != PAPI_OK) {
	ret = PAPI_event_code_to_name (papieventlist[n], eventname);
	return GPTLerror ("%s: failure adding event:%s Error was: %s\n", 
			  thisfunc, eventname, PAPI_strerror (ret));
      }
    }
  }

  // Start the event set.  It will only be read from now on--never stopped
  if ((ret = PAPI_start (EventSet[t])) != PAPI_OK)
    return GPTLerror ("%s: failed to start event set: %s\n", thisfunc, PAPI_strerror (ret));

  return 0;
}

/*
** GPTL_PAPIstart: Start the PAPI counters (actually they are just read).  
**   Called from GPTLstart.
**
** Input args:  
**   t: thread number
**
** Output args: 
**   aux: struct containing the counters
**
** Return value: 0 (success) or GPTLerror (failure)
*/
int GPTL_PAPIstart (const int t, Papistats *aux)
{
  int ret;
  int n;
  static const char *thisfunc = "GPTL_PAPIstart";
  
  // If no events are to be counted just return
  if (npapievents == 0)
    return 0;

  // Read the counters
  if ((ret = PAPI_read (EventSet[t], papicounters[t])) != PAPI_OK)
    return GPTLerror ("%s: %s\n", thisfunc, PAPI_strerror (ret));

  // Store the counter values.  When PAPIstop is called, the counters
  // will again be read, and differenced with the values saved here.
  for (n = 0; n < npapievents; n++)
    aux->last[n] = papicounters[t][n];
  
  return 0;
}

/*
** GPTL_PAPIstop: Stop the PAPI counters (actually they are just read).  
**   Called from GPTLstop.
**
** Input args:
**   t: thread number
**
** Input/output args: 
**   aux: struct containing the counters
**
** Return value: 0 (success) or GPTLerror (failure)
*/
int GPTL_PAPIstop (const int t, Papistats *aux)
{
  int ret;          // return code from PAPI lib calls
  int n;            // loop index
  long_long delta;  // change in counters from previous read
  static const char *thisfunc = "GPTL_PAPIstop";

  // If no events are to be counted just return
  if (npapievents == 0)
    return 0;

  // Read the counters
  if ((ret = PAPI_read (EventSet[t], papicounters[t])) != PAPI_OK)
    return GPTLerror ("%s: %s\n", thisfunc, PAPI_strerror (ret));
  
  /* 
  ** Accumulate the difference since timer start in aux.
  ** Negative accumulation can happen when multiplexing is enabled, so don't
  ** set count to BADCOUNT in that case.
  */
  for (n = 0; n < npapievents; n++) {
#ifdef DEBUG
    printf ("%s: event %d counter value is %ld\n", thisfunc, n, (long) papicounters[t][n]);
#endif
    delta = papicounters[t][n] - aux->last[n];
    if ( ! is_multiplexed && delta < 0)
      aux->accum[n] = BADCOUNT;
    else
      aux->accum[n] += delta;
  }
  return 0;
}

/*
** GPTL_PAPIprstr: Print the descriptive string for all enabled PAPI events.
**   Called from GPTLpr.
**
** Input args: 
**   fp: file descriptor
*/
void GPTL_PAPIprstr (FILE *fp)
{
  int n;
  
  for (n = 0; n < nevents; n++) {
    fprintf (fp, " %16.16s", pr_event[n].event.str8);

    // Test on < 0 says it's a PAPI preset
    if (persec && pr_event[n].event.counter < 0)
      fprintf (fp, " e6_/_sec");
  }
}

/*
** GPTL_PAPIpr: Print PAPI counter values for all enabled events, including
**   derived events. Called from GPTLpr.
**
** Input args: 
**   fp: file descriptor
**   aux: struct containing the counters
*/
void GPTL_PAPIpr (FILE *fp, const Papistats *aux, const int t, const int count, const double wcsec)
{
  const char *intfmt   = " %8ld";
  const char *floatfmt = " %8.2e";

  int n;              // loop index
  int numidx;         // index pointer to appropriated (derived) numerator
  int denomidx;       // index pointer to appropriated (derived) denominator
  double val;         // value to be printed
  static const char *thisfunc = "GPTL_PAPIpr";

  for (n = 0; n < nevents; n++) {
    numidx = pr_event[n].numidx;
    if (pr_event[n].denomidx > -1) {      // derived event
      denomidx = pr_event[n].denomidx;

#ifdef DEBUG
      printf ("%s: derived event: numidx=%d denomidx=%d values = %ld %ld\n", 
	      thisfunc, numidx, denomidx, (long) aux->accum[numidx], (long) aux->accum[denomidx]);
#endif
      // Protect against divide by zero
      if (aux->accum[denomidx] > 0)
	val = (double) aux->accum[numidx] / (double) aux->accum[denomidx];
      else
	val = 0.;
      fprintf (fp, floatfmt, val);

    } else {                               // Raw PAPI event

#ifdef DEBUG
      printf ("%s: raw event: numidx=%d value = %ld\n", 
	      thisfunc, numidx, (long) aux->accum[numidx]);
#endif
      if (aux->accum[numidx] < PRTHRESH)
	fprintf (fp, intfmt, (long) aux->accum[numidx]);
      else
	fprintf (fp, floatfmt, (double) aux->accum[numidx]);

      if (persec) {
	if (wcsec > 0.)
	  fprintf (fp, " %8.2f", aux->accum[numidx] * 1.e-6 / wcsec);
	else
	  fprintf (fp, " %8.2f", 0.);
      }
    }
  }
}

// PAPIprintenabled: Print list of enabled timers
void GPTL_PAPIprintenabled (FILE *fp)
{
  int n, nn;
  PAPI_event_info_t info;           // returned from PAPI_get_event_info
  char eventname[PAPI_MAX_STR_LEN]; // returned from PAPI_event_code_to_name

  if (nevents > 0) {
    fprintf (fp, "Description of printed events (PAPI and derived):\n");
    for (n = 0; n < nevents; n++) {
      if (strncmp (pr_event[n].event.namestr, "GPTL", 4) == 0) {
	fprintf (fp, "  %s\n", pr_event[n].event.namestr);
      } else {
	nn = pr_event[n].event.counter;
	if (PAPI_get_event_info (nn, &info) == PAPI_OK) {
	  fprintf (fp, "  %s\n", info.short_descr);
	  fprintf (fp, "  %s\n", info.note);
	}
      }
    }
    fprintf (fp, "\n");

    fprintf (fp, "PAPI events enabled (including those required for derived events):\n");
    for (n = 0; n < npapievents; n++)
      if (PAPI_event_code_to_name (papieventlist[n], eventname) == PAPI_OK)
	fprintf (fp, "  %s\n", eventname);
    fprintf (fp, "\n");
  }
}  

/*
** GPTL_PAPIadd: Accumulate PAPI counters. Called from add.
**
** Input/Output args: 
**   auxout: auxout = auxout + auxin
**
** Input args:
**   auxin: counters to be summed into auxout
*/
void GPTL_PAPIadd (Papistats *auxout, const Papistats *auxin)
{
  int n;
  
  for (n = 0; n < npapievents; n++)
    if (auxin->accum[n] == BADCOUNT || auxout->accum[n] == BADCOUNT)
      auxout->accum[n] = BADCOUNT;
    else
      auxout->accum[n] += auxin->accum[n];
}

// PAPIfinalize: finalization routine must be called from single-threaded
//   region. Free all malloc'd space
void GPTL_PAPIfinalize (int maxthreads)
{
  int t;   /* thread index */
  int ret; /* return code */

  for (t = 0; t < maxthreads; t++) {
    ret = PAPI_stop (EventSet[t], papicounters[t]);
    free (papicounters[t]);
    ret = PAPI_cleanup_eventset (EventSet[t]);
    ret = PAPI_destroy_eventset (&EventSet[t]);
  }

  free (EventSet);
  free (papicounters);

  // Reset initial values
  npapievents = 0;
  nevents = 0;
  is_multiplexed = false;
  persec = true;
  enable_multiplexing = false;
  verbose = false;
}

/*
** GPTL_PAPIquery: return current PAPI counter info. Return into a long for best
**   compatibility possibilities with Fortran.
**
** Input args:
**   aux:       struct containing the counters
**   ncounters: max number of counters to return
**
** Output args:
**   papicounters_out: current value of PAPI counters
*/
void GPTL_PAPIquery (const Papistats *aux, long long *papicounters_out, int ncounters)
{
  int n;

  if (ncounters > 0) {
    for (n = 0; n < ncounters && n < npapievents; n++) {
      papicounters_out[n] = (long long) aux->accum[n];
    }
  }
}

/*
** GPTL_PAPIget_eventvalue: return current value for an enabled event.
**
** Input args:
**   eventname: event name to check (whether derived or raw PAPI counter)
**   aux:       struct containing the counter(s) for the event
**
** Output args:
**   value: current value of the event
**
** Return value: 0 (success) or GPTLerror (failure)
*/
int GPTL_PAPIget_eventvalue (const char *eventname, const Papistats *aux, double *value)
{
  int n;        // loop index through enabled events
  int numidx;   // numerator index into papicounters
  int denomidx; // denominator index into papicounters
  static const char *thisfunc = "GPTL_PAPIget_eventvalue";

  for (n = 0; n < nevents; ++n) {
    if (STRMATCH (eventname, pr_event[n].event.namestr)) {
      numidx = pr_event[n].numidx;
      if (pr_event[n].denomidx > -1) {  // derived event
	denomidx = pr_event[n].denomidx;
	if (aux->accum[denomidx] > 0)   // protect against divide by zero
	  *value = (double) aux->accum[numidx] / (double) aux->accum[denomidx];
	else
	  *value = 0.;
      } else {        // Raw PAPI event
	*value = (double) aux->accum[numidx];
      }
      break;
    }
  }
  if (n == nevents)
    return GPTLerror ("%s: event %s not enabled\n", thisfunc, eventname);
  return 0;
}

// GPTL_PAPIis_multiplexed: return status of whether events are being multiplexed
bool GPTL_PAPIis_multiplexed () {return is_multiplexed;}

/*
** The following functions are publicly available
*/
void read_counters1000 ()
{
  int i;
  int ret;
  long_long counters[MAX_AUX];

#pragma unroll(10)
  for (i = 0; i < 1000; ++i) {
    ret = PAPI_read (EventSet[0], counters);
  }
  return;
}

/*
** GPTLevent_name_to_code: convert a string to a PAPI code
** or derived event code.
**
** Input arguments:
**   arg: string to convert
**
** Output arguments:
**   code: PAPI or GPTL derived code
**
** Return value: 0 (success) or GPTLerror (failure)
*/
int GPTLevent_name_to_code (const char *name, int *code)
{
  int ret;   // return code
  int n;     // loop over derived entries
  static const char *thisfunc = "GPTLevent_name_to_code";

  // First check derived events
  for (n = 0; n < nderivedentries; ++n) {
    if (STRMATCH (name, derivedtable[n].namestr)) {
      *code = derivedtable[n].counter;
      return 0;
    }
  }

  // Next check PAPI events--note that PAPI must be initialized before the
  // name_to_code function can be invoked.
  if ((ret = PAPI_is_initialized ()) == PAPI_NOT_INITED) {
    printf ("%s: PAPI not initialized. Calling PAPI_library_init()...\n", thisfunc);
    if ((ret = GPTL_PAPIlibraryinit ()) < 0)
      return GPTLerror ("%s: GPTL_PAPIlibraryinit failure\n", thisfunc);
  }

  if ((PAPI_event_name_to_code ((char *) name, code)) != PAPI_OK)
    return GPTLerror ("%s: PAPI_event_name_to_code failure\n", thisfunc);

  return 0;
}

/*
** GPTLevent_code_to_name: convert a string to a PAPI code
** or derived event code.
**
** Input arguments:
**   code: event code (PAPI or derived)
**
** Output arguments:
**   name: string corresponding to code
**
** Return value: 0 (success) or GPTLerror (failure)
*/
int GPTLevent_code_to_name (const int code, char *name)
{
  int ret;
  int n;
  static const char *thisfunc = "GPTLevent_code_to_name";

  // First check derived events
  for (n = 0; n < nderivedentries; ++n) {
    if (code == derivedtable[n].counter) {
      strcpy (name, derivedtable[n].namestr);
      return 0;
    }
  }

  // Next check PAPI events--note that PAPI must be initialized before the
  // code_to_name function can be invoked.
  if ((ret = PAPI_is_initialized ()) == PAPI_NOT_INITED) {
    printf ("%s: PAPI not initialized. Calling PAPI_library_init()...\n", thisfunc);
    if ((ret = GPTL_PAPIlibraryinit ()) < 0)
      return GPTLerror ("%s: GPTL_PAPIlibraryinit failure\n", thisfunc);
  }

  if (PAPI_event_code_to_name (code, name) != PAPI_OK)
    return GPTLerror ("%s: PAPI_event_code_to_name failure\n", thisfunc);

  return 0;
}

int GPTLget_npapievents (void) {return npapievents;}

#ifdef __cplusplus
}
#endif
