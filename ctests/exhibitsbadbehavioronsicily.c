#include <sys/time.h>      /* gettimeofday */
#include <stdio.h>
#include <stdlib.h>

static time_t refsec = -1;
double mygettimeofday (void);

int main ()
{
  struct timeval tp1;         /* argument to gettimeofday */
  double delta;
  double prv;
  double timestamp;
  int iter;

  gettimeofday (&tp1, 0);
  refsec = tp1.tv_sec;
  prv = mygettimeofday ();

  for (iter = 0; ; iter++) {
      timestamp = mygettimeofday ();
      delta = timestamp - prv;
      if (delta < 0. || delta > 100.) {
	  printf ("bad delta=%g at iter=%d prv=%g timestamp=%g\n", 
		  delta, iter, prv, timestamp);
	  exit (1);
      } else {
	  if (iter % 10000000 == 0) 
	      printf ("iter %d ok delta=%g\n", iter, delta);
      }
      prv = timestamp;
  }
}

double mygettimeofday ()
{
  struct timeval tp;
  (void) gettimeofday (&tp, 0);
  return (tp.tv_sec - refsec) + 1.e-6*tp.tv_usec;
}
  
