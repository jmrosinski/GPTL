/*
$Id: private.h,v 1.11 2004-10-16 20:22:14 rosinski Exp $
*/

#include "gpt.h"

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

#define STRMATCH(X,Y) (strcmp((X),(Y)) == 0)
#define MAX_CHARS 15

typedef enum {false = 0, true = 1} bool;

typedef struct {
  long last_sec;
  long last_usec;
  long accum_sec;
  long accum_usec;
  float max;
  float min;
  float overhead;
} Wallstats;

typedef struct {
  long last_utime;
  long last_stime;
  long accum_utime;
  long accum_stime;
} Cpustats;

typedef struct {
  long some_compilers_dont_allow_empty_structs;
} Auxstats;
  
typedef struct TIMER {
  char name[MAX_CHARS+1];
  bool onflg;
  int depth;
  long count;
  Wallstats wall;
  Cpustats cpu;
  /*
  ** For later when hooked to PAPI or PCL
  *  Auxstats aux; 
  */
  struct TIMER *next;
} Timer;

typedef struct {
  const Option option;
  const char *name;
  const char *str;
  bool enabled;
} Settings;

typedef struct {
  int nument;
  Timer **entries;
} Hashtable;

/* Function prototypes */

extern int GPTerror (const char *, ...);
extern void GPTset_abort_on_error (bool val);
extern int threadinit (int *);            /* initialize threading environment */
extern int get_thread_num (int *);        /* determine thread number */
