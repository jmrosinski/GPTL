#include <stdio.h>

static int junk;
extern void callutil10times ();
extern void callutil100times ();
extern void util ();
extern void A ();
extern void B ();
extern void C ();

void callsubs (int niter)
{
  printf ("callsubs calling callutil10times\n");
  callutil10times ();
  printf ("callsubs calling callutil100times\n");
  callutil100times ();
  printf ("callsubs calling A\n");
  A();
  printf ("callsubs calling util\n");
  util ();
}

void callutil10times ()
{
  int n;

  printf ("callutil10times calling util\n");
  for (n = 0; n < 10; ++n) 
    util ();
}

void callutil100times ()
{
  int n;

  printf ("callutil100times calling util\n");
  for (n = 0; n < 100; ++n)
    util ();
}

void util () 
{
  junk = 11;
}

void A ()
{
  printf ("A calling B\n");
  B ();
}

void B ()
{
  printf ("B calling C\n");
  C ();
}

void C ()
{
  printf ("C calling util\n");
  util ();
  printf ("C calling callutil10times\n");
  callutil10times ();
}
