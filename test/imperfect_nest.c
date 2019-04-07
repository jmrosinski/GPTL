#include <gptl.h>
int main ()
{
  int ret;
  
  ret = GPTLinitialize ();
  ret = GPTLstart ("main");
  ret = GPTLstart ("firstsub");
  ret = GPTLstart ("secondsub");
  ret = GPTLstop ("firstsub");
  ret = GPTLstop ("secondsub");
  ret = GPTLstop ("main");

  ret = GPTLpr (0);
}

  
