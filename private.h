/*
$Id: private.h,v 1.16 2004-10-31 17:32:38 rosinski Exp $
*/

#include <stdio.h>
#include "gpt.h"

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

#define STRMATCH(X,Y) (strcmp((X),(Y)) == 0)
#define MAX_CHARS 15
#define MAX_AUX 8

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
  long long last[MAX_AUX];
  long long accum[MAX_AUX];
} Papistats;
  
typedef struct TIMER {
  char name[MAX_CHARS+1];
  bool onflg;
  unsigned int depth;
  unsigned long count;
  Wallstats wall;
  Cpustats cpu;
  Papistats aux; 
  struct TIMER *next;
} Timer;

typedef struct {
  unsigned int nument;
  Timer **entries;
} Hashtable;

/* Function prototypes */

extern int GPTerror (const char *, ...);
extern void GPTset_abort_on_error (bool val);
extern void *GPTallocate (const int);
extern int threadinit (int *, int *);          /* initialize threading environment */
extern void threadfinalize (void);             /* finalize threading environment */
extern int get_thread_num (int *, int *);      /* determine thread number */

#ifdef HAVE_PAPI
extern int GPT_PAPIsetoption (const int, const int);
extern int GPT_PAPIinitialize (const int);
extern int GPT_PAPIstart (const int, Papistats *);
extern int GPT_PAPIstop (const int, Papistats *);
extern void GPT_PAPIprstr (FILE *);
extern void GPT_PAPIpr (FILE *, const Papistats *);
extern void GPT_PAPIadd (Papistats *, const Papistats *);
#endif
