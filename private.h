/*
$Id: private.h,v 1.4 2004-10-14 19:25:54 rosinski Exp $
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
  //  char *name = "wall";
  //  char *printstr = "Wallclock max       min";
  long last_sec;
  long last_usec;
  long accum_sec;
  long accum_usec;
  float max;
  float min;
  float overhead;
} Wallstats;

typedef struct {
  //  char *name = "cpu";
  //  char *printstr = "Usr       sys       usr+sys";
  long last_utime;
  long last_stime;
  long accum_utime;
  long accum_stime;
} Cpustats;

typedef struct {
} Auxstats;
  
typedef struct TIMER {
  char *name;
  int depth;
  long count;
  int *max_depth;         // max depth in timer tree (for indentation)
  int *current_depth;     // current depth in timer tree (for indentation)
  int indent_level;
  Wallstats wall;
  Cpustats cpu;
  Auxstats aux;
  bool gather_wall;
  bool gather_cpu;
  bool gather_aux;
  bool onflg;
  struct TIMER *next;
} Timer;

typedef struct {
  Option option;
  char *str;
  bool enabled;
} Settings;

// Function prototypes

extern int GPTerror (const char *, ...);
