#include "../gptl.h"
#include <stdio.h>

int main ()
{
  int ret;
  int i, code;
  long long pc[1]; /* papi counters */
  double sum;

  printf ("testpapi: Testing PAPI interface...\n");

  printf ("Testing getting event code for PAPI_TOT_CYC...\n");
  if ((ret = GPTLevent_name_to_code ("PAPI_TOT_CYC", &code)) != 0) {
    printf ("Failure\n");
    return 2;
  }
  printf ("Success\n");

  printf ("Testing GPTLsetoption(PAPI_TOT_CYC,1)...\n");
  if (GPTLsetoption (code, 1) != 0) {
    printf ("Failure\n");
    return 3;
  }
  printf ("Success\n");

  printf ("Testing GPTLinitialize\n");
  if ((ret = GPTLinitialize ()) != 0) {
    printf ("Failure\n");
    return 3;
  }
  printf ("Success\n");

  if ((ret = GPTLstart ("sum")) != 0) {
    printf ("Unexpected failure from GPTLstart(\"sum\")\n");
    return 3;
  }
  sum = 0.;
  for (i = 0; i < 1000000; ++i) 
    sum += (double) i;
  if ((ret = GPTLstop ("sum")) != 0) {
    printf ("Unexpected failure from GPTLstop(\"sum\")\n");
    return 3;
  }

  printf ("Testing GPTLquerycounters...\n");
  if (GPTLquerycounters ("sum", 0, pc) != 0) {
    printf ("Failure\n");
    return 4;
  }
  printf ("sum=%g\n",sum);
  if (pc[0] < 1 || pc[0] > 1.e8) {
    printf ("Suspicious PAPI_TOT_CYC value=%ld for 1e6 additions\n",pc[0]);
    return 5;
  } else {
    printf ("Success\n");
  }
  printf ("testpapi: all tests successful\n");
  return 0;
}
