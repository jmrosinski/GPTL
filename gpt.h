/*
$Id: gpt.h,v 1.2 2001-01-01 19:34:05 rosinski Exp $
*/

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

typedef enum {false = 0, true = 1} Boolean;

/*
** Function prototypes
*/

extern int add_new_thread (void);
extern int get_cpustamp (long *, long *);
extern int get_thread_num (void);
extern int GPTerror (const char *, ...);
extern int GPTinitialize (void);
extern int GPTpr (int);
extern int GPTreset (void);
extern int GPTsetoption (OptionName, Boolean);
extern int GPTstamp (double *, double *, double *);
extern int GPTstart (char *);
extern int GPTstop (char *);
extern char *GPTpclstr (int);
