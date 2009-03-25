#include "../gptl.h"

int main ()
{
  X *x;
  Y *y;

  x = new (X);
  x.func (1.2);
  x.func (1);

  delete (x);
}
