/*
$Id: private.h,v 1.1 2000-12-27 06:10:31 rosinski Exp $
*/

#if ( defined THREADED_OMP )
#include <omp.h>
#elif ( defined THREADED_PTHREADS )
#include <pthread.h>
#endif

#ifdef HAVE_PCL
#include <pcl.h>
#else

/*
** Dummy up pcl stuff if library unavailable
*/

typedef int PCL_CNT_TYPE;
typedef int PCL_FP_CNT_TYPE;
typedef int PCL_DESCR_TYPE;
#define PCL_MODE_USER       -1
#define PCL_L1DCACHE_MISS   -1
#define PCL_L2CACHE_MISS    -1
#define PCL_CYCLES          -1
#define PCL_ELAPSED_CYCLES  -1
#define PCL_FP_INSTR        -1
#define PCL_LOADSTORE_INSTR -1
#define PCL_INSTR           -1
#define PCL_STALL           -1
#define PCL_COUNTER_MAX      1
#define PCL_SUCCESS          0
extern int PCLread (PCL_DESCR_TYPE, PCL_CNT_TYPE *, PCL_CNT_TYPE *, int);
#endif

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

#define STRMATCH(X,Y) (strcmp((X),(Y)) == 0)
#define MAX_CHARS 15
#define AMBIGUOUS -1
#define MAX_THREADS 128

typedef enum {false = 0, true = 1} Boolean;

/*
** User specifiable options.  The values must match their counterparts in header.inc
** Also, we must have pcl_start < all valid pcl values < pcl_end.
** To add a new PCL counter: 
** 1) add the new entry to OptionName below.
** 2) add the appropriate array entry for possible_event[] to t_initialize.c.
** 3) add the appropriate code to the "switch" construct in t_initialize.c
*/

typedef enum {
  usrsys               = 1,
  wall                 = 2,
  pcl_start            = 3,   /* bogus entry delimits start of PCL stuff */
#ifdef HAVE_PCL
  pcl_l1dcache_miss    = 4,
  pcl_l2cache_miss     = 5,
  pcl_cycles           = 6,
  pcl_elapsed_cycles   = 7,
  pcl_fp_instr         = 8,
  pcl_loadstore_instr  = 9,
  pcl_instr            = 10,
  pcl_stall            = 11,
#endif
  pcl_end              = 12,  /* bogus entry delimits end of PCL stuff */
} OptionName;

struct node {
  char name[MAX_CHARS+1];
  
  int indent_level;        /* indentation level of timer */

  long last_utime;         /* user time from most recent call */
  long last_stime;         /* system time from most recent call */
  long last_wtime_sec;     /* wallclock seconds from most recent call */
  long last_wtime_usec;    /* wallclock microseconds from most recent call */

  long accum_utime;        /* accumulated user time */
  long accum_stime;        /* accumulated system time */
  long accum_wtime_sec;    /* accumulated wallclock seconds */
  long accum_wtime_usec;   /* accumulated wallclock microseconds */

  float max_wtime;         /* maximum wallclock time for each start-stop */
  float min_wtime;         /* minimum wallclock time for each start-stop */

  PCL_CNT_TYPE last_pcl_result[PCL_COUNTER_MAX];
  PCL_CNT_TYPE accum_pcl_result[PCL_COUNTER_MAX];

  Boolean onflg;           /* true => timer is currently on */
  long count;              /* number of calls to t_start for this timer */

  struct node *next;       /* next timer in the linked list */
};

#if ( defined THREADED_OMP )
extern omp_lock_t lock;
#elif ( defined THREADED_PTHREADS )
extern pthread_mutex_t t_mutex;
extern pthread_t *threadid;
#endif
