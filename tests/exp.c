#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include "../gpt.h"
main()
{
  char name[9];
  int i,n;
  double sum;

  GPTinitialize(); 
  GPTsetoption (usrsys, true);
  GPTstart("total");
  for (n = 0; n < 98; n++) {
    if (n < 10 ) {
      sprintf(name,"%s%1d","loop_0",n+1);
    } else {
      sprintf(name,"%s%2d","loop_",n+1);
    }
    GPTstart(name);
    GPTstop(name);
  }
  GPTstart("loop_99");
  for (n = 0; n < 100; n++) {
    GPTstart("loop_100");
    sum = 0.;
    for (i = 0; i < 100000; i++) {
      sum += log((double) i+1);
    }
    printf("sum=%e\n",sum);
    GPTstop("loop_100");
  }
  GPTstop("loop_99");
  GPTstart("usleep");
  for (n = 0; n < 1000; n++) {
    GPTstart("usleep1000");
    usleep(1000);
    GPTstop("usleep1000");
  }  
  GPTstop("usleep");
  GPTstop("total");
  GPTpr(0);
  exit(0);
}

