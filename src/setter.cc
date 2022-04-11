#include "config.h" // Must be first include.
#include <string.h>

#include "private.h"
#include "main.h"
#include "thread.h"
#include "once.h"

// GPTLenable: enable timers 
int GPTLenable (void)
{
  gptlmain::disabled = false;
  return (0);
}

// GPTLdisable: disable timers
int GPTLdisable (void)
{
  gptlmain::disabled = true;
  return (0);
}

// GPTLreset: reset all timers to 0
// Return value: 0 (success) or GPTLerror (failure)
int GPTLreset (void)
{
  int t;
  Timer *ptr;
  static const char *thisfunc = "GPTLreset";

  if ( ! gptlmain::initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);

  for (t = 0; t < thread::nthreads; t++) {
    for (ptr = gptlmain::timers[t]; ptr; ptr = ptr->next) {
      ptr->onflg = false;
      ptr->count = 0;
      memset (&ptr->wall, 0, sizeof (ptr->wall));
      memset (&ptr->cpu, 0, sizeof (ptr->cpu));
#ifdef HAVE_PAPI
      memset (&ptr->aux, 0, sizeof (ptr->aux));
#endif
#ifdef ENABLE_NESTEDOMP
      ptr->major = -1;
      ptr->minor = -1;
#endif
    }
  }

  if (once::verbose)
    printf ("%s: accumulators for all timers set to zero\n", thisfunc);

  return 0;
}

// GPTLreset_timer: reset a timer to 0
// Return value: 0 (success) or GPTLerror (failure)
int GPTLreset_timer (const char *name)
{
  int t;
  Timer *ptr;
  unsigned int indx; // hash index
  int namelen;
  static const char *thisfunc = "GPTLreset_timer";

  if ( ! gptlmain::initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);

  if (thread::get_thread_num () != 0)
    return GPTLerror ("%s: Must be called by the master thread\n", thisfunc);

  namelen = strlen (name);
  indx = gptlmain::genhashidx (name, namelen);
  for (t = 0; t < thread::nthreads; ++t) {
    ptr = gptlmain::getentry (gptlmain::hashtable[t], name, indx);
    if (ptr) {
      ptr->onflg = false;
      ptr->count = 0;
      memset (&ptr->wall, 0, sizeof (ptr->wall));
      memset (&ptr->cpu, 0, sizeof (ptr->cpu));
#ifdef HAVE_PAPI
      memset (&ptr->aux, 0, sizeof (ptr->aux));
#endif
#ifdef ENABLE_NESTEDOMP
      ptr->major = -1;
      ptr->minor = -1;
#endif
    }
  }
  return 0;
}

