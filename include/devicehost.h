// longest timer name allowed (probably safe to just change)
#define MAX_CHARS 63
// Output counts less than PRTHRESH will be printed as integers
#define PRTHRESH 1000000L

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

#ifdef __cplusplus
extern "C" {
#endif
extern int GPTLerror (const char *, ...);                  // print error msg and return
#ifdef __cplusplus
}
#endif
