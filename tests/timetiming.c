#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <sys/times.h>
#include "../gpt.h"
main()
{
  int n;
  struct tms buf;

  GPTsetoption (GPTcpu, 0);
  GPTsetoption (GPTwall, 1);

  GPTinitialize ();

  for (n = 0; n < 100000; n++) {
    GPTstart ("times");
    (void) times (&buf);
    GPTstop ("times");
  }
  GPTpr (0);
  return 0;
}

