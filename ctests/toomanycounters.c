#include "../gptl.h"
#include "../private.h"
#include <papi.h>
int main ()
{
  int ret;
  int i, code;
  const int nevents = 9;
  char event[9][13] = {"PAPI_TOT_CYC","PAPI_TOT_INS","PAPI_L1_TCM","PAPI_FP_INS",
		       "PAPI_FAD_INS","PAPI_FML_INS","PAPI_L1_DCA","PAPI_BR_INS",
		       "PAPI_L1_DCM"};

  ret = PAPI_library_init (PAPI_VER_CURRENT);
  if (ret != PAPI_VER_CURRENT) {
    printf ("ret, PAPI_VER_CURRENT=%d %d\n", ret, PAPI_VER_CURRENT);
    printf ("Failure to initialize PAPI library\n");
    return 1;
  }

  if (nevents > MAX_AUX) {
    printf ("toomanycounters: Trying to enable too many PAPI counters...\n");
    for (i = 0; i < nevents; i++) {
      ret = PAPI_event_name_to_code (event[i], &code);
      if (GPTLsetoption (code, 1) != 0) {
	printf ("Too many at index %d (MAX_AUX=%d)\n", i, MAX_AUX);
	return 0;
      }
    }
    printf ("Failure to fail!\n");
    return 1;
  } else {
    printf ("toomanycounters: can't test anything: MAX_AUX=%d needs to be smaller than nevents=%d\n",
	    MAX_AUX, nevents);
  }
  printf ("Success\n");
  return 0;
}
