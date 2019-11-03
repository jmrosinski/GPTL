#include "config.h"
#include "gptl.h"
#include <papi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main (int argc, char **argv)
{
  int ret;
  int i;
  int code;                  // papi event code
  int nevents = 0;           // number of papi events
  int c;                     // for parsing arg list
  int EventSet = PAPI_NULL;  // Event set needed by PAPI lib
  double pc[1];              // papi counters
  double sum;
  char eventname[PAPI_MAX_STR_LEN] = "PAPI_TOT_CYC"; // Default papi counter
  char eventsave[PAPI_MAX_STR_LEN];                  // Saved papi counter
  char *thisfunc = argv[0];

  printf ("testpapi: Testing PAPI interface...\n");

  // Parse arg list: only count the last one
  while ((c = getopt (argc, argv, "e:")) != -1) {
    switch (c) {
    case 'e':
      // Convert name to code
      strcpy (eventname, optarg);
      break;
    default:
      printf ("unknown option %c\n", c);
      return -1;
    }
  }

  // Verify code_to_name and inverse are working properly
  strcpy (eventsave, eventname);
  if ((ret = GPTLevent_name_to_code (eventname, &code)) != PAPI_OK) {
    printf ("No code found for event %s\n", optarg);
    printf ("PAPI_strerror says: %s\n", PAPI_strerror (ret));
    return -1;
  }
  
  // Add the event
  printf ("%s: testing GPTLsetoption(code for %s,1)...\n", thisfunc, eventname);
  if (GPTLsetoption (code, 1) != 0) {
    printf ("Failure\n");
    return 3;
  }

  printf ("Testing that code=%d translates to name=%s\n", code, eventname);
  if ((ret = GPTLevent_code_to_name (code, eventname)) != 0) {
    printf ("Failure\n");
    return 2;
  }
  if (strcmp (eventname, eventsave) == 0) {
    printf ("Success: eventname=%s\n", eventname);
  } else {
    printf ("Failure: expected %s got %s\n", eventsave, eventname);
    return 2;
  }

  printf ("%s: Testing GPTLinitialize\n", thisfunc);
  if ((ret = GPTLinitialize ()) != 0) {
    printf ("Failure\n");
    return 3;
  }

  printf ("%s: testing GPTLstart\n", thisfunc);
  if ((ret = GPTLstart ("sum")) != 0) {
    printf ("Unexpected failure from GPTLstart(\"sum\")\n");
    return 3;
  }

  sum = 0.;
  for (i = 0; i < 1000000; ++i) 
    sum += (double) i;

  printf ("%s: testing GPTLstop\n", thisfunc);
  if ((ret = GPTLstop ("sum")) != 0) {
    printf ("Unexpected failure from GPTLstop(\"sum\")\n");
    return 3;
  }

  printf ("%s: testing GPTLget_eventvalue...\n", thisfunc);
  if (GPTLget_eventvalue ("sum", eventname, 0, pc) != 0) {
    printf ("Failure\n");
    return 4;
  }
  printf ("Success: counter=%f\n", pc[0]);

  if (strcmp (eventname, "PAPI_TOT_CYC") == 0) {
    printf ("%s: testing reasonableness of PAPI counters...\n", thisfunc);
    if (pc[0] < 1 || pc[0] > 1.e8) {
      printf ("Suspicious PAPI_TOT_CYC value=%f for 1e6 additions\n", pc[0]);
      return 5;
    } else {
      printf ("Success\n");
    }
  }
  if ((ret = GPTLpr (0)) != 0) {
    printf ("%s: bad return from GPTLpr()\n", thisfunc);
    return 6;
  }
  printf ("%s: All tests successful\n", thisfunc);
  return 0;
}
