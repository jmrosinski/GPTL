/*
$Id: gpt.h,v 1.6 2004-10-16 00:03:44 rosinski Exp $
*/

typedef enum {
  GPTwall           = 1,
  GPTcpu            = 2,
  GPTabort_on_error = 3,
  GPTother          = 4
} Option;

/*
** Function prototypes
*/

extern int GPTsetoption (const Option, const int);
extern int GPTinitialize (void);
extern int GPTstart (const char *);
extern int GPTstop (const char *);
extern int GPTstamp (double *, double *, double *);
extern int GPTpr (const int);
extern int GPTreset (void);
