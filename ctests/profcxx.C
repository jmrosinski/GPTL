#include "../gptl.h"
#include "myclasses.h"

int main ()
{
  X *x;
  Y *y;
  int ret;

  ret = GPTLinitialize ();
  ret = GPTLstart ("total");

  x = new (X);
  x->func (1.2);
  x->func (1);
  delete (x);

  y = new (Y);
  y->func (1.2);
  y->func (1);
  delete (y);

  ret = GPTLstop ("total");
  ret = GPTLpr (0);
}

