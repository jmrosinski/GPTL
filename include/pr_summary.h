namespace {
  typedef struct {
    unsigned long totcalls;  // number of calls to the region across threads and tasks
#ifdef HAVE_PAPI
    double papimax[MAX_AUX]; // max counter value across threads, tasks
    double papimin[MAX_AUX]; // max counter value across threads, tasks
    int papimax_p[MAX_AUX];  // task producing papimax
    int papimax_t[MAX_AUX];  // thread producing papimax
    int papimin_p[MAX_AUX];  // task producing papimin
    int papimin_t[MAX_AUX];  // thread producing papimin
#endif
    unsigned int notstopped; // number of ranks+threads for whom the timer is ON
    unsigned int tottsk;     // number of tasks which invoked this region
    float wallmax;           // max time across threads, tasks
    float wallmin;           // min time across threads, tasks
    float mean;              // accumulated mean
    float m2;                // from Chan, et. al.
    int wallmax_p;           // task producing wallmax
    int wallmax_t;           // thread producing wallmax
    int wallmin_p;           // task producing wallmin
    int wallmin_t;           // thread producing wallmin
    char name[MAX_CHARS+1];  // timer name
  } Global;

  extern "C" {
    void get_threadstats (int, char *, Timer **, Global *);
    Timer *getentry_slowway (Timer *, char *);
  }
}
