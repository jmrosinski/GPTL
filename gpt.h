/*
$Id: gpt.h,v 1.1 2000-12-27 06:10:31 rosinski Exp $
*/

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
