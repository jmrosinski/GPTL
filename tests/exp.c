#include <math.h>
#include <sys/time.h>     /* gettimeofday */
#include <unistd.h>       /* gettimeofday */
#include <stdio.h>
#include "../gpt.h"
main()
{
  char name[10];
  int i,n;
  double sum;

  struct timeval tp1;      /* argument to gettimeofday */

  GPTsetoption (GPTcpu, 1);
  GPTsetoption (GPTwall, 1);

  GPTinitialize ();

  GPTstart("total");

  for (n = 0; n < 999; n++) {
    printf ("n=%d\n", n);
    sprintf (name,"%s%4.4d","loop_",n+1);
    GPTstart(name);
    gettimeofday (&tp1, 0);   /* for overhead est. */
    GPTstop(name);
  }
  GPTstart("loop_99");
  for (n = 0; n < 100; n++) {
    GPTstart("loop_100");
    sum = 0.;
    for (i = 0; i < 100000; i++) {
      sum += log(i+1.) + sqrt(i*1.);
    }
    printf("sum=%e\n",sum);
    GPTstop("loop_100");
  }
  GPTstop("loop_99");
  GPTstart("usleep");
  for (n = 0; n < 1000; n++) {
    GPTstart("usleep1000");
    /*    usleep(1000); */
    GPTstop("usleep1000");
  }  
  GPTstop("usleep");
  GPTstop("total");
  GPTpr(0);
  return 0;
}

