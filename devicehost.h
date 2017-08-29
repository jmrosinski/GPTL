/*
** $Id$
**
** Author: Jim Rosinski
**
** Contains definitions shared between CPU and GPU
*/

/* longest timer name allowed (probably safe to just change) */
#define MAX_CHARS 63
#ifdef ENABLE_GPU
#define DEFAULT_MAXTHREADS_GPU 1280
#define DEFAULT_TABLE_SIZE_GPU 63
#define DEFAULT_MAXTIMERS_GPU 20
#define MAXPARENT 3

typedef struct {
  int count;
  long long accum;
  long long max;
  long long min;
  unsigned int nparent;
  char name[MAX_CHARS+1];
  char parentname[MAXPARENT][MAX_CHARS+1];
} Gpustats;

#endif
