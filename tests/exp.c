#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include "../header.h"
main()
{
  char name[9];
  int i,n;
  double sum;

  t_initialize(); 
  t_setoption (usrsys, true);
  t_start("total");
  for (n = 0; n < 98; n++) {
    if (n < 10 ) {
      sprintf(name,"%s%1d","loop_0",n+1);
    } else {
      sprintf(name,"%s%2d","loop_",n+1);
    }
    t_start(name);
    t_stop(name);
  }
  t_start("loop_99");
  for (n = 0; n < 100; n++) {
    t_start("loop_100");
    sum = 0.;
    for (i = 0; i < 100000; i++) {
      sum += log((double) i+1);
    }
    printf("sum=%e\n",sum);
    t_stop("loop_100");
  }
  t_stop("loop_99");
  t_start("usleep");
  for (n = 0; n < 1000; n++) {
    t_start("usleep1000");
    usleep(1000);
    t_stop("usleep1000");
  }  
  t_stop("usleep");
  t_stop("total");
  t_pr(0);
  exit(0);
}

