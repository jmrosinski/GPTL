#include <math.h>
#include <sys/time.h>     /* gettimeofday */
#include <unistd.h>       /* gettimeofday */
#include <stdio.h>
#include "../gptl.h"

int main(int argc, char **argv)
{
  int niter = 1000;
  int iter;

  GPTLsetoption (GPTLcpu, 0);
  GPTLsetoption (GPTLwall, 1);
  GPTLsetoption (GPTLabort_on_error, 1);
  GPTLsetoption (GPTLparentchild, 1);

  GPTLinitialize ();

  for (iter = 0; iter < 10; iter++) {
    GPTLstart ("A");
    GPTLstart ("B");
    GPTLstart ("C");

    GPTLstart ("D");
    GPTLstop ("D");

    if (iter > 0) {
      GPTLstart ("DD:Misplaced_child_of_C");
      GPTLstart ("EE:child_of_DD");
      GPTLstop ("EE:child_of_DD");
      GPTLstop ("DD:Misplaced_child_of_C");
    }

    GPTLstop ("C");
    GPTLstart ("CC");
    GPTLstop ("CC");

    GPTLstop ("B");
    GPTLstart ("BB");
    GPTLstop ("BB");

    GPTLstop ("A");
    GPTLstart ("AA");
    GPTLstart ("BB");
    GPTLstop ("BB");
    GPTLstart ("BBB");
    GPTLstop ("BBB");
    GPTLstop ("AA");
  }

  GPTLpr (0);
  GPTLfinalize ();

  return 0;
}
