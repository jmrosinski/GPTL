#include <stdio.h>
#include <stdlib.h>
#include "../gptl.h"

void do_nothing (char *);

int main ()
{
  char *stuff;
  (void) GPTLprint_memusage ("startup");
  stuff = (char *) malloc (104857600);
  do_nothing (stuff);
  (void) GPTLprint_memusage ("after malloc 100 MB");
  return 0;
}

void do_nothing (char *stuff)
{
  stuff[0] = 'x';
  stuff[104857599 ] = 'x';
}
