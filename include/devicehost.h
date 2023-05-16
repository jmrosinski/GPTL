// longest timer name allowed (probably safe to just change)
// Must be at least 16 to hold auto-profiled name, and 9 to hold "GPTL_ROOT"
#define MAX_CHARS 63

// Output counts less than PRTHRESH will be printed as integers
#define PRTHRESH 1000000L

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif
