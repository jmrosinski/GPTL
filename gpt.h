/*
$Id: gpt.h,v 1.7 2004-10-19 03:16:18 rosinski Exp $
*/

typedef enum {
  GPTwall           = 1,
  GPTcpu            = 2,
  GPTabort_on_error = 3,
  GPTother          = 4
} GPTOption;

/*
** Function prototypes
*/

extern int GPTsetoption (const GPTOption, const int);
extern int GPTinitialize (void);
extern int GPTfinalize (void);
extern int GPTstart (const char *);
extern int GPTstop (const char *);
extern int GPTstamp (double *, double *, double *);
extern int GPTpr (const int);
extern int GPTreset (void);
