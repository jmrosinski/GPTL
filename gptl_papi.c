#ifdef HAVE_PAPI

#include <papi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if ( defined THREADED_OMP )
#include <omp.h>
#elif ( defined THREADED_PTHREADS )
#include <pthread.h>
#endif

#include "private.h"

typedef struct {
  int counter;      /* PAPI counter */
  char *counterstr; /* PAPI counter as string */
  char *prstr;      /* print string for output timers (16 chars) */
  char *str;        /* descriptive print string (more descriptive than prstr) */
} Papientry;

/* Mapping of PAPI counters to short and long printed strings */

static Papientry papitable [] = {
  {PAPI_L1_DCM, "PAPI_L1_DCM", "L1 Dcache miss  ", "Level 1 data cache misses"},
  {PAPI_L1_ICM, "PAPI_L1_ICM", "L1 Icache miss  ", "Level 1 instruction cache misses"},
  {PAPI_L2_DCM, "PAPI_L2_DCM", "L2 Dcache miss  ", "Level 2 data cache misses"},
  {PAPI_L2_ICM, "PAPI_L2_ICM", "L2 Icache miss  ", "Level 2 instruction cache misses"},
  {PAPI_L3_DCM, "PAPI_L3_DCM", "L3 Dcache miss  ", "Level 3 data cache misses"},
  {PAPI_L3_ICM, "PAPI_L3_ICM", "L3 Icache miss  ", "Level 3 instruction cache misses"},
  {PAPI_L1_TCM, "PAPI_L1_TCM", "L1 cache miss   ", "Level 1 total cache misses"},
  {PAPI_L2_TCM, "PAPI_L2_TCM", "L2 cache miss   ", "Level 2 total cache misses"},
  {PAPI_L3_TCM, "PAPI_L3_TCM", "L3 cache miss   ", "Level 3 total cache misses"},
  {PAPI_CA_SNP, "PAPI_CA_SNP", "Snoops          ", "Snoops          "},
  {PAPI_CA_SHR, "PAPI_CA_SHR", "PAPI_CA_SHR     ", "Request for shared cache line (SMP)"},
  {PAPI_CA_CLN, "PAPI_CA_CLN", "PAPI_CA_CLN     ", "Request for clean cache line (SMP)"},
  {PAPI_CA_INV, "PAPI_CA_INV", "PAPI_CA_INV     ", "Request for cache line Invalidation (SMP)"},
  {PAPI_CA_ITV, "PAPI_CA_ITV", "PAPI_CA_ITV     ", "Request for cache line Intervention (SMP)"},
  {PAPI_L3_LDM, "PAPI_L3_LDM", "L3 load misses  ", "Level 3 load misses"},
  {PAPI_L3_STM, "PAPI_L3_STM", "L3 store misses ", "Level 3 store misses"},
  {PAPI_BRU_IDL,"PAPI_BRU_IDL","PAPI_BRU_IDL    ", "Cycles branch units are idle"},
  {PAPI_FXU_IDL,"PAPI_FXU_IDL","PAPI_FXU_IDL    ", "Cycles integer units are idle"},
  {PAPI_FPU_IDL,"PAPI_FPU_IDL","PAPI_FPU_IDL    ", "Cycles floating point units are idle"},
  {PAPI_LSU_IDL,"PAPI_LSU_IDL","PAPI_LSU_IDL    ", "Cycles load/store units are idle"},
  {PAPI_TLB_DM,"PAPI_TLB_DM", "Data TLB misses ", "Data translation lookaside buffer misses"},
  {PAPI_TLB_IM,"PAPI_TLB_IM", "Inst TLB misses ", "Instr translation lookaside buffer misses"},
  {PAPI_TLB_TL,"PAPI_TLB_TL", "Tot TLB misses  ", "Total translation lookaside buffer misses"},
  {PAPI_L1_LDM,"PAPI_L1_LDM", "L1 load misses  ", "Level 1 load misses"},
  {PAPI_L1_STM,"PAPI_L1_STM", "L1 store misses ", "Level 1 store misses"},
  {PAPI_L2_LDM,"PAPI_L2_LDM", "L2 load misses  ", "Level 2 load misses"},
  {PAPI_L2_STM,"PAPI_L2_STM", "L2 store misses ", "Level 2 store misses"},
  {PAPI_BTAC_M,"PAPI_BTAC_M", "BTAC miss       ", "BTAC miss"},
  {PAPI_PRF_DM,"PAPI_PRF_DM", "PAPI_PRF_DM     ", "Prefetch data instruction caused a miss"},
  {PAPI_L3_DCH,"PAPI_L3_DCH", "L3 DCache Hit   ", "Level 3 Data Cache Hit"},
  {PAPI_TLB_SD,"PAPI_TLB_SD", "PAPI_TLB_SD     ", "Xlation lookaside buffer shootdowns (SMP)"},
  {PAPI_CSR_FAL,"PAPI_CSR_FAL","PAPI_CSR_FAL    ", "Failed store conditional instructions"},
  {PAPI_CSR_SUC,"PAPI_CSR_SUC","PAPI_CSR_SUC    ", "Successful store conditional instructions"},
  {PAPI_CSR_TOT,"PAPI_CSR_TOT","PAPI_CSR_TOT    ", "Total store conditional instructions"},
  {PAPI_MEM_SCY,"PAPI_MEM_SCY","Cyc Stalled Mem ", "Cycles Stalled Waiting for Memory Access"},
  {PAPI_MEM_RCY,"PAPI_MEM_RCY","Cyc Stalled MemR", "Cycles Stalled Waiting for Memory Read"},
  {PAPI_MEM_WCY,"PAPI_MEM_WCY","Cyc Stalled MemW", "Cycles Stalled Waiting for Memory Write"},
  {PAPI_STL_ICY,"PAPI_STL_ICY","Cyc no InstrIss ", "Cycles with No Instruction Issue"},
  {PAPI_FUL_ICY,"PAPI_FUL_ICY","Cyc Max InstrIss", "Cycles with Maximum Instruction Issue"},
  {PAPI_STL_CCY,"PAPI_STL_CCY","Cyc No InstrComp", "Cycles with No Instruction Completion"},
  {PAPI_FUL_CCY,"PAPI_FUL_CCY","Cyc Max InstComp", "Cycles with Maximum Instruction Completion"},
  {PAPI_HW_INT,"PAPI_HW_INT", "HW interrupts   ", "Hardware interrupts"},
  {PAPI_BR_UCN,"PAPI_BR_UCN", "Uncond br instr ", "Unconditional branch instructions executed"},
  {PAPI_BR_CN,"PAPI_BR_CN",  "Cond br instr ex", "Conditional branch instructions executed"},
  {PAPI_BR_TKN,"PAPI_BR_TKN", "Cond br instr tk", "Conditional branch instructions taken"},
  {PAPI_BR_NTK,"PAPI_BR_NTK", "Cond br instrNtk", "Conditional branch instructions not taken"},
  {PAPI_BR_MSP,"PAPI_BR_MSP", "Cond br instrMPR", "Conditional branch instructions mispred"},
  {PAPI_BR_PRC,"PAPI_BR_PRC", "Cond br instrCPR", "Conditional branch instructions corr. pred"},
  {PAPI_FMA_INS,"PAPI_FMA_INS","FMA instr comp  ", "FMA instructions completed"},
  {PAPI_TOT_IIS,"PAPI_TOT_IIS","Total instr iss ", "Total instructions issued"},
  {PAPI_TOT_INS,"PAPI_TOT_INS","Total instr ex  ", "Total instructions executed"},
  {PAPI_INT_INS,"PAPI_INT_INS","Int instr ex    ", "Integer instructions executed"},
  {PAPI_FP_INS, "PAPI_FP_INS", "FP instr ex     ", "Floating point instructions executed"},
  {PAPI_LD_INS,"PAPI_LD_INS", "Load instr ex   ", "Load instructions executed"},
  {PAPI_SR_INS,"PAPI_SR_INS", "Store instr ex  ", "Store instructions executed"},
  {PAPI_BR_INS,"PAPI_BR_INS", "br instr ex     ", "Total branch instructions executed"},
  {PAPI_VEC_INS,"PAPI_VEC_INS","Vec/SIMD instrEx", "Vector/SIMD instructions executed"},
  {PAPI_RES_STL,"PAPI_RES_STL","Cyc proc stalled", "Cycles processor is stalled on resource"},
  {PAPI_FP_STAL,"PAPI_FP_STAL","Cyc any FP stall", "Cycles any FP units are stalled"},
  {PAPI_TOT_CYC,"PAPI_TOT_CYC","Total cycles    ", "Total cycles"},
  {PAPI_LST_INS,"PAPI_LST_INS","Tot L/S inst ex ", "Total load/store inst. executed"},
  {PAPI_SYC_INS,"PAPI_SYC_INS","Sync. inst. ex  ", "Sync. inst. executed"},
  {PAPI_L1_DCH,"PAPI_L1_DCH", "L1 D Cache Hit  ", "L1 D Cache Hit"},
  {PAPI_L2_DCH,"PAPI_L2_DCH", "L2 D Cache Hit  ", "L2 D Cache Hit"},
  {PAPI_L1_DCA,"PAPI_L1_DCA", "L1 D Cache Acc  ", "L1 D Cache Access"},
  {PAPI_L2_DCA,"PAPI_L2_DCA", "L2 D Cache Acc  ", "L2 D Cache Access"},
  {PAPI_L3_DCA,"PAPI_L3_DCA", "L3 D Cache Acc  ", "L3 D Cache Access"},
  {PAPI_L1_DCR,"PAPI_L1_DCR", "L1 D Cache Read ", "L1 D Cache Read"},
  {PAPI_L2_DCR,"PAPI_L2_DCR", "L2 D Cache Read ", "L2 D Cache Read"},
  {PAPI_L3_DCR,"PAPI_L3_DCR", "L3 D Cache Read ", "L3 D Cache Read"},
  {PAPI_L1_DCW,"PAPI_L1_DCW", "L1 D Cache Write", "L1 D Cache Write"},
  {PAPI_L2_DCW,"PAPI_L2_DCW", "L2 D Cache Write", "L2 D Cache Write"},
  {PAPI_L3_DCW,"PAPI_L3_DCW", "L3 D Cache Write", "L3 D Cache Write"},
  {PAPI_L1_ICH,"PAPI_L1_ICH", "L1 I cache hits ", "L1 instruction cache hits"},
  {PAPI_L2_ICH,"PAPI_L2_ICH", "L2 I cache hits ", "L2 instruction cache hits"},
  {PAPI_L3_ICH,"PAPI_L3_ICH", "L3 I cache hits ", "L3 instruction cache hits"},
  {PAPI_L1_ICA,"PAPI_L1_ICA", "L1 I cache acc  ", "L1 instruction cache accesses"},
  {PAPI_L2_ICA,"PAPI_L2_ICA", "L2 I cache acc  ", "L2 instruction cache accesses"},
  {PAPI_L3_ICA,"PAPI_L3_ICA", "L3 I cache acc  ", "L3 instruction cache accesses"},
  {PAPI_L1_ICR,"PAPI_L1_ICR", "L1 I cache reads", "L1 instruction cache reads"},
  {PAPI_L2_ICR,"PAPI_L2_ICR", "L2 I cache reads", "L2 instruction cache reads"},
  {PAPI_L3_ICR,"PAPI_L3_ICR", "L3 I cache reads", "L3 instruction cache reads"},
  {PAPI_L1_ICW,"PAPI_L1_ICW", "L1 I cache write", "L1 instruction cache writes"},
  {PAPI_L2_ICW,"PAPI_L2_ICW", "L2 I cache write", "L2 instruction cache writes"},
  {PAPI_L3_ICW,"PAPI_L3_ICW", "L3 I cache write", "L3 instruction cache writes"},
  {PAPI_L1_TCH,"PAPI_L1_TCH", "L1 cache hits   ", "L1 total cache hits"},
  {PAPI_L2_TCH,"PAPI_L2_TCH", "L2 cache hits   ", "L2 total cache hits"},
  {PAPI_L3_TCH,"PAPI_L3_TCH", "L3 cache hits   ", "L3 total cache hits"},
  {PAPI_L1_TCA,"PAPI_L1_TCA", "L1 cache access ", "L1 total cache accesses"},
  {PAPI_L2_TCA,"PAPI_L2_TCA", "L2 cache access ", "L2 total cache accesses"},
  {PAPI_L3_TCA,"PAPI_L3_TCA", "L3 cache access ", "L3 total cache accesses"},
  {PAPI_L1_TCR,"PAPI_L1_TCR", "L1 cache reads  ", "L1 total cache reads"},
  {PAPI_L2_TCR,"PAPI_L2_TCR", "L2 cache reads  ", "L2 total cache reads"},
  {PAPI_L3_TCR,"PAPI_L3_TCR", "L3 cache reads  ", "L3 total cache reads"},
  {PAPI_L1_TCW,"PAPI_L1_TCW", "L1 cache writes ", "L1 total cache writes"},
  {PAPI_L2_TCW,"PAPI_L2_TCW", "L2 cache writes ", "L2 total cache writes"},
  {PAPI_L3_TCW,"PAPI_L3_TCW", "L3 cache writes ", "L3 total cache writes"},
  {PAPI_FML_INS,"PAPI_FML_INS","FM ins          ", "FM ins"},
  {PAPI_FAD_INS,"PAPI_FAD_INS","FA ins          ", "FA ins"},
  {PAPI_FDV_INS,"PAPI_FDV_INS","FD ins          ", "FD ins"},
  {PAPI_FSQ_INS,"PAPI_FSQ_INS","FSq ins         ", "FSq ins"},
  {PAPI_FNV_INS,"PAPI_FNV_INS","Finv ins        ", "Finv ins"},
  {PAPI_FP_OPS,"PAPI_FP_OPS", "FP ops executed ", "Floating point operations executed"}
};

static const int nentries = sizeof (papitable) / sizeof (Papientry);
static Papientry eventlist[MAX_AUX];     /* list of PAPI events to be counted */
static Papientry propeventlist[MAX_AUX]; /* list of PAPI events hoped to be counted */
static int nevents = 0;                  /* number of events: initialize to 0 */ 
static int nprop = 0;                    /* number of hoped events: initialize to 0 */ 
static int *EventSet;                    /* list of events to be counted by PAPI */
static long_long **papicounters;         /* counters return from PAPI */
static char papiname[PAPI_MAX_STR_LEN];  /* returned from PAPI_event_code_to_name */
static const int BADCOUNT = -999999;     /* Set counters to this when they are bad */
static int GPTLoverheadindx = -1;        /* index into counters array */
static long_long *readoverhead;          /* overhead due to reading PAPI counters */
static bool is_multiplexed = false;      /* whether multiplexed (always start false)*/
static bool narrowprint = false;         /* only use 8 digits not 16 for counter prints */
const static bool enable_multiplexing = true; /* whether to try multiplexing */

/* Function prototypes */

static int create_and_start_events (const int);

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

int GPTL_PAPIsetoption (const int counter,  /* PAPI counter (or option) */
			const int val)      /* true or false for enable or disable */
{
  int n;   /* loop index */
  int ret; /* return code */
  
  /* Just return if the flag says disable an option, because default is off */

  if ( ! val)
    return 0;

  /*
  ** Loop through table looking for counter. If found, add the entry to the
  ** list of "proposed events".  Won't know till init time whether the event
  ** is available on this arch.
  */

  for (n = 0; n < nentries; n++) {
    if (counter == papitable[n].counter) {
      if (nprop+1 > MAX_AUX) {
	return GPTLerror ("GPTL_PAPIsetoption: Event %s is too many\n", 
			  papitable[n].str);
      } else {
	propeventlist[nprop].counter    = counter;
	propeventlist[nprop].counterstr = papitable[n].counterstr;
	propeventlist[nprop].prstr      = papitable[n].prstr;
	propeventlist[nprop].str        = papitable[n].str;
	printf ("GPTL_PAPIsetoption: will attempt to enable event %s\n", 
		propeventlist[nprop].str);
	++nprop;
      }
      return 0;
    }
  }

  /*
  ** Now check native events
  */
  
  if ((ret = PAPI_event_code_to_name (counter, papiname)) == PAPI_OK) {
    if (nprop+1 > MAX_AUX) {
      return GPTLerror ("GPTL_PAPIsetoption: Event %d is too many\n", counter);
    } else {
      propeventlist[nprop].counter    = counter;
      
      /*
      ** Only one name known for native events, that which is in papiname
      */

      propeventlist[nprop].counterstr = GPTLallocate (strlen (papiname)+1);
      propeventlist[nprop].prstr      = GPTLallocate (strlen (papiname)+1);
      propeventlist[nprop].str        = GPTLallocate (strlen (papiname)+1);

      strcpy (propeventlist[nprop].counterstr, papiname);
      strcpy (propeventlist[nprop].prstr, papiname);
      strcpy (propeventlist[nprop].str, papiname);

      printf ("GPTL_PAPIsetoption: will attempt to enable event %s\n", 
	      propeventlist[nprop].str);
      ++nprop;
    }
    return 0;
  }

  /*
  ** Finally, check for option which is not an actual counter
  */

  if (counter == GPTLnarrowprint) {
    narrowprint = (bool) val;
    return 0;
  }
  return GPTLerror ("GPTL_PAPIsetoption: counter %d does not exist\n", counter);
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
 
int GPTL_PAPIinitialize (const int maxthreads)  /* number of threads */
{
  int ret;       /* return code */
  int n;         /* loop index */
  int counter;   /* PAPI counter */
  int t;         /* thread index */
  int *rc;       /* array of return codes from create_and_start_events */
  bool badret;   /* true if any bad return codes were found */

  /* 
  ** PAPI_library_init needs to be called before ANY other PAPI routine.
  ** Check that the user hasn't already called PAPI_library_init before
  ** invoking it.
  */

  if ((ret = PAPI_is_initialized ()) == PAPI_NOT_INITED) {
    if ((ret = PAPI_library_init (PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      return GPTLerror ("GPTL_PAPIinitialize: PAPI_library_init failure:%s\n",
			PAPI_strerror (ret));
  }

  /* PAPI_thread_init needs to be called if threading enabled */

#if ( defined THREADED_OMP )
  if (PAPI_thread_init ((unsigned long (*)(void)) (omp_get_thread_num)) != PAPI_OK)
    return GPTLerror ("GPTL_PAPIinitialize: PAPI_thread_init failure\n");
#elif ( defined THREADED_PTHREADS )
  if (PAPI_thread_init ((unsigned long (*)(void)) (pthread_self)) != PAPI_OK)
    return GPTLerror ("GPTL_PAPIinitialize: PAPI_thread_init failure\n");
#endif

  /* allocate and initialize static local space */

  EventSet     = (int *)        GPTLallocate (maxthreads * sizeof (int));
  papicounters = (long_long **) GPTLallocate (maxthreads * sizeof (long_long *));
  readoverhead = (long_long *)  GPTLallocate (maxthreads * sizeof (long_long));

  for (t = 0; t < maxthreads; t++) {
    EventSet[t] = PAPI_NULL;
    papicounters[t] = (long_long *) GPTLallocate (MAX_AUX * sizeof (long_long));
    readoverhead[t] = -1;
  }

  /* 
  ** Loop over events set by earlier calls to GPTL_PAPIsetoption. For the
  ** events which can be counted on this architecture, fill in the values
  ** (array "eventlist")
  */

  for (n = 0; n < nprop; n++) {
    counter = propeventlist[n].counter;
    if (PAPI_query_event (counter) != PAPI_OK) {
      (void) PAPI_event_code_to_name (counter, papiname);
      return GPTLerror ("GPTL_PAPIinitialize: event %s not available on this arch\n", papiname);
    } else {
      if (nevents+1 > MAX_AUX) {
	(void) PAPI_event_code_to_name (counter, papiname);
	return GPTLerror ("GPTL_PAPIinitialize: Event %s is too many\n", papiname);
      } else {
	if (counter == PAPI_TOT_CYC)
	  GPTLoverheadindx = nevents;

	eventlist[nevents].counter    = counter;
	eventlist[nevents].counterstr = propeventlist[n].counterstr;
	eventlist[nevents].prstr      = propeventlist[n].prstr;
	eventlist[nevents].str        = propeventlist[n].str;
	printf ("GPTL_PAPIinitialize: event %s enabled\n", eventlist[nevents].str);
	++nevents;
      }
    }
  }

  /* Event starting apparently must be within a threaded loop. */

  if (nevents > 0) {
    rc = (int *) GPTLallocate (maxthreads * sizeof (int));

#pragma omp parallel for private (t)

    for (t = 0; t < maxthreads; t++)
      rc[t] = create_and_start_events (t);
  
    badret = false;
    for (t = 0; t < maxthreads; t++)
      if (rc[t] < 0)
	badret = true;
    
    free (rc);

    if (badret)
      return -1;
  }

  return 0;
}

/*
** create_and_start_events: Create and start the PAPI eventset.  File-local.
**   Threaded routine to create the "event set" (PAPI terminology) and start
**   the counters. This is only done once, and is called from GPTL_PAPIinitialize 
** 
** Input args: 
**   t: thread number
**
** Return value: 0 (success) or GPTLerror (failure)
*/

static int create_and_start_events (const int t)  /* thread number */
{
  int ret;
  int n;
  int i;
  long_long counters1[MAX_AUX];  /* Temp counter for estimating PAPI_read overhead  */
  long_long counters2[MAX_AUX];  /* Temp counter for estimating PAPI_read overhead  */

  /* Create the event set */

  if ((ret = PAPI_create_eventset (&EventSet[t])) != PAPI_OK)
    return GPTLerror ("create_and_start_events: failure creating eventset: %s\n", 
		      PAPI_strerror (ret));

  /* Add requested events to the event set */

  for (n = 0; n < nevents; n++) {
    if ((ret = PAPI_add_event (EventSet[t], eventlist[n].counter)) != PAPI_OK) {
      printf ("%s\n", PAPI_strerror (ret));
      printf ("create_and_start_events: failure adding event:%s\n", 
	      eventlist[n].str);

      if (enable_multiplexing) {
	printf ("Trying multiplexing...\n");
	is_multiplexed = true;
	break;
      } else
	return GPTLerror ("enable_multiplexing is false: giving up\n");
    }
  }

  if (is_multiplexed) {

    /* Cleanup the eventset for multiplexing */

    if ((ret = PAPI_cleanup_eventset (EventSet[t])) != PAPI_OK)
      return GPTLerror ("create_and_start_events: %s\n", PAPI_strerror (ret));
    
    if ((ret = PAPI_destroy_eventset (&EventSet[t])) != PAPI_OK)
      return GPTLerror ("create_and_start_events: %s\n", PAPI_strerror (ret));

    if ((ret = PAPI_create_eventset (&EventSet[t])) != PAPI_OK)
      return GPTLerror ("create_and_start_events: failure creating eventset: %s\n", 
			PAPI_strerror (ret));

    if ((ret = PAPI_multiplex_init ()) != PAPI_OK)
      return GPTLerror ("create_and_start_events: failure from PAPI_multiplex_init%s\n", 
			PAPI_strerror (ret));

    if ((ret = PAPI_set_multiplex (EventSet[t])) != PAPI_OK)
      return GPTLerror ("create_and_start_events: failure from PAPI_set_multiplex: %s\n", 
			PAPI_strerror (ret));

    for (n = 0; n < nevents; n++) {
      if ((ret = PAPI_add_event (EventSet[t], eventlist[n].counter)) != PAPI_OK)
	return GPTLerror ("create_and_start_events: failure adding event:%s\n"
			  "  Error was: %s\n", eventlist[n].str, PAPI_strerror (ret));
    }
  }

    /* Start the event set.  It will only be read from now on--never stopped */

  if ((ret = PAPI_start (EventSet[t])) != PAPI_OK)
    return GPTLerror ("create_and_start_events: failed to start event set: %s\n", PAPI_strerror (ret));

  /* Estimate overhead of calling PAPI_read, to be used later in printing */

  if (GPTLoverheadindx > -1) {
    if ((ret = PAPI_read (EventSet[t], counters1)) != PAPI_OK)
      return GPTLerror ("create_and_start_events: %s\n", PAPI_strerror (ret));

    for (i = 0; i < 10; ++i) {
      if ((ret = PAPI_read (EventSet[t], counters2)) != PAPI_OK)
	return GPTLerror ("create_and_start_events: %s\n", PAPI_strerror (ret));
      if ((ret = PAPI_read (EventSet[t], counters2)) != PAPI_OK)
	return GPTLerror ("create_and_start_events: %s\n", PAPI_strerror (ret));
      if ((ret = PAPI_read (EventSet[t], counters2)) != PAPI_OK)
	return GPTLerror ("create_and_start_events: %s\n", PAPI_strerror (ret));
      if ((ret = PAPI_read (EventSet[t], counters2)) != PAPI_OK)
	return GPTLerror ("create_and_start_events: %s\n", PAPI_strerror (ret));
      if ((ret = PAPI_read (EventSet[t], counters2)) != PAPI_OK)
	return GPTLerror ("create_and_start_events: %s\n", PAPI_strerror (ret));
      if ((ret = PAPI_read (EventSet[t], counters2)) != PAPI_OK)
	return GPTLerror ("create_and_start_events: %s\n", PAPI_strerror (ret));
      if ((ret = PAPI_read (EventSet[t], counters2)) != PAPI_OK)
	return GPTLerror ("create_and_start_events: %s\n", PAPI_strerror (ret));
      if ((ret = PAPI_read (EventSet[t], counters2)) != PAPI_OK)
	return GPTLerror ("create_and_start_events: %s\n", PAPI_strerror (ret));
      if ((ret = PAPI_read (EventSet[t], counters2)) != PAPI_OK)
	return GPTLerror ("create_and_start_events: %s\n", PAPI_strerror (ret));
      if ((ret = PAPI_read (EventSet[t], counters2)) != PAPI_OK)
	return GPTLerror ("create_and_start_events: %s\n", PAPI_strerror (ret));
    }
    
    readoverhead[t] = 0.01 * (counters2[GPTLoverheadindx] - counters1[GPTLoverheadindx]);
  }

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

int GPTL_PAPIstart (const int t,          /* thread number */
		    Papistats *aux)       /* struct containing PAPI stats */
{
  int ret;  /* return code from PAPI lib calls */
  int n;    /* loop index */
  
  /* If no events are to be counted just return */

  if (nevents == 0)
    return 0;

  /* Read the counters */

  if ((ret = PAPI_read (EventSet[t], papicounters[t])) != PAPI_OK)
    return GPTLerror ("GPTL_PAPIstart: %s\n", PAPI_strerror (ret));

  /* 
  ** Store the counter values.  When GPTL_PAPIstop is called, the counters
  ** will again be read, and differenced with the values saved here.
  */

  for (n = 0; n < nevents; n++)
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

int GPTL_PAPIstop (const int t,         /* thread number */
		   Papistats *aux)      /* struct containing PAPI stats */
{
  int ret;          /* return code from PAPI lib calls */
  int n;            /* loop index */
  long_long delta;  /* change in counters from previous read */

  /* If no events are to be counted just return */

  if (nevents == 0)
    return 0;

  /* Read the counters */

  if ((ret = PAPI_read (EventSet[t], papicounters[t])) != PAPI_OK)
    return GPTLerror ("GPTL_PAPIstop: %s\n", PAPI_strerror (ret));
  
  /* 
  ** Accumulate the difference since timer start in aux.
  ** If negative accumulation has occurred (unfortunately this can and does
  ** happen, especially on AIX), store a flag value (BADCOUNT)
  */

  for (n = 0; n < nevents; n++) {
    delta = papicounters[t][n] - aux->last[n];
    if (delta < 0)
      aux->accum[n] = BADCOUNT;
    else if (aux->accum[n] != BADCOUNT)
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

void GPTL_PAPIprstr (FILE *fp,                          /* file descriptor */
		     const bool overheadstatsenabled)   /* whether to print overhead stats*/
{
  int n;
  
  if (narrowprint) {
    for (n = 0; n < nevents; n++)
      fprintf (fp, "%8.8s ", &eventlist[n].counterstr[5]); /* 5 => lop off "PAPI_" */
    if (overheadstatsenabled && GPTLoverheadindx > -1)
      fprintf (fp, "OH (cyc) ");
  } else {
    for (n = 0; n < nevents; n++)
      fprintf (fp, "%16.16s ", eventlist[n].prstr);
    if (overheadstatsenabled && GPTLoverheadindx > -1)
      fprintf (fp, "Overhead (cyc)   ");
  }
}

/*
** GPTL_PAPIpr: Print PAPI counter values for all enabled events. Called from
**   GPTLpr.
**
** Input args: 
**   fp: file descriptor
**   aux: struct containing the counters
*/

void GPTL_PAPIpr (FILE *fp,                          /* file descriptor to write to */
		  const Papistats *aux,              /* stats to write */
		  const int t,                       /* thread number */
		  const int count,                   /* number of invocations */
		  const bool overheadstatsenabled)   /* whether to print overhead stats*/
{
  int n;
  long_long overhead;  /* overhead due to PAPI_read */
  
  for (n = 0; n < nevents; n++) {
    if (narrowprint) {
      if (aux->accum[n] < 1000000)
	fprintf (fp, "%8ld ", (long) aux->accum[n]);
      else
	fprintf (fp, "%8.2e ", (double) aux->accum[n]);
    } else {
      if (aux->accum[n] < 1000000)
	fprintf (fp, "%16ld ", (long) aux->accum[n]);
      else
	fprintf (fp, "%16.10e ", (double) aux->accum[n]);
    }
  }

  /* 
  ** Print overhead estimate. Note similarity of overhead calc. to
  ** overhead calc. in gptl.c.
  */

  if (overheadstatsenabled && GPTLoverheadindx > -1) {
    overhead = count * 2 * readoverhead[t];
    if (narrowprint) {
      if (overhead < 1000000)
	fprintf (fp, "%8ld ", (long) overhead);
      else
	fprintf (fp, "%8.2e ", (double) overhead);
    } else {
      if (overhead < 1000000)
	fprintf (fp, "%8ld ", (long) overhead);
      else
	fprintf (fp, "%8.2e ", (double) overhead);
    }
  }
}

/*
** GPTL_PAPIprintenabled: Print list of enabled timers
**
** Input args:
**   fp: file descriptor
*/

void GPTL_PAPIprintenabled (FILE *fp)
{
  int n;

  fprintf (fp, "PAPI events enabled:\n");
  for (n = 0; n < nevents; n++)
    fprintf (fp, "  %s\n", eventlist[n].str);
  fprintf (fp, "\n");
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

void GPTL_PAPIadd (Papistats *auxout,      /* output struct */
		   const Papistats *auxin) /* input struct */
{
  int n;
  
  for (n = 0; n < nevents; n++)
    if (auxin->accum[n] == BADCOUNT || auxout->accum[n] == BADCOUNT)
      auxout->accum[n] = BADCOUNT;
    else
      auxout->accum[n] += auxin->accum[n];

  /* Overhead calcs */

  if (auxin->accum_cycles == BADCOUNT || auxout->accum_cycles == BADCOUNT)
    auxout->accum_cycles = BADCOUNT;
  else
    auxout->accum_cycles += auxin->accum_cycles;
}

/*
** GPTL_PAPIfinalize: finalization routine must be called from single-threaded
**   region. Free all malloc'd space
*/

void GPTL_PAPIfinalize (int maxthreads)
{
  int t;

  for (t = 0; t < maxthreads; t++) {
    free (papicounters[t]);
  }

  free (EventSet);
  free (papicounters);
  free (readoverhead);

  /* Reset initial values */

  nevents = 0;
  nprop = 0;
  GPTLoverheadindx = -1;
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

void GPTL_PAPIquery (const Papistats *aux,
		     long *papicounters_out,
		     int ncounters)
{
  int n;

  if (ncounters > 0) {
    for (n = 0; n < ncounters && n < nevents; n++) {
      papicounters_out[n] = (long long) aux->accum[n];
    }
  }
}

/*
** GPTL_PAPIis_multiplexed: return status of whether events are being multiplexed
*/

bool GPTL_PAPIis_multiplexed ()
{
  return is_multiplexed;
}

/*
** The following functions are publicly available
*/

/*
** GPTL_PAPIprinttable:  Print table of PAPI native counters. Not all are
**   necessarily available on this architecture. This is the one routine 
**   in this file which is user-visible.
*/

void GPTL_PAPIprinttable ()
{
  int n;

  for (n = 0; n < nentries; n++)
    printf ("%d %s\n", papitable[n].counter, papitable[n].str);
}

/*
** GPTL_PAPIname2id: convert a PAPI event name in string form to an int
**
** Input args:
**   name: PAPI event name
**
** Return value: event index (success) or 77 (error)
*/

int GPTL_PAPIname2id (const char *name)
{
  int i;
  for (i = 0; i < nentries; i++) {
  if (strcmp (name, papitable[i].counterstr) == 0)
    return papitable[i].counter;
  }
  printf ("GPTL_PAPIname2id: %s not found\n", name);
  return 77; /* successful return is a negative number */
}

#else

/*
** "Should not be called" entry points for publicly available GPTL_PAPI routines
*/

#include <stdio.h>

void GPTL_PAPIprinttable ()
{
  printf ("PAPI not enabled: GPTL_PAPIprinttable does nothing\n");
  return;
}

int GPTL_PAPIname2id (const char *name, int nc)
{
  printf ("PAPI not enabled: GPTL_PAPIname2id should not be called\n");
  return 77;
}

#endif  /* HAVE_PAPI */
