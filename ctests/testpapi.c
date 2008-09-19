#include "../gptl.h"
#include <papi.h>
int main ()
{
  int ret;
  int i, code;
  long long pc[1]; /* papi counters */
  double sum;

  printf ("testpapi: Testing PAPI interface...\n");
  ret = PAPI_library_init (PAPI_VER_CURRENT);
  if (ret != PAPI_VER_CURRENT) {
    printf ("ret, PAPI_VER_CURRENT=%d %d\n", ret, PAPI_VER_CURRENT);
    printf ("Failure to initialize PAPI library\n");
    return 1;
  }
  ret = PAPI_event_name_to_code ("PAPI_TOT_CYC", &code);
  if (ret != 0) {
    printf ("Failure from PAPI_event_name_to_code()\n");
    return 2;
  }
  printf ("Testing enabling PAPI_TOT_CYC...\n");
  if (GPTLsetoption (code, 1) != 0) {
    printf ("Failure\n");
    return 3;
  }
  printf ("Success\n");
  printf ("Testing GPTLinitialize\n");
  if (GPTLinitialize () != 0) {
    printf ("Failure\n");
    return 3;
  }
  printf ("Success\n");

  ret = GPTLstart ("sum");
  sum = 0.;
  for (i = 0; i < 1000000; ++i) 
    sum += (double) i;
  ret = GPTLstop ("sum");

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
  return 0;
}
