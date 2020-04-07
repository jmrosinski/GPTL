int GPTLthreadid = -1;   // This probably isn't needed

static int threadinit (void)
{
  static const char *thisfunc = "threadinit";

  if (nthreads != -1)
    return GPTLerror ("GPTL: Unthreaded %s: MUST only be called once", thisfunc);

  nthreads = 0;
  maxthreads = 1;
  return 0;
}

void threadfinalize ()
{
  GPTLthreadid = -1;
}

static inline int get_thread_num ()
{
#ifdef HAVE_PAPI
  static const char *thisfunc = "get_thread_num";
  /*
  ** When HAVE_PAPI is true, if 1 or more PAPI events are enabled,
  ** create and start an event set for the new thread.
  */
  if (GPTLthreadid == -1 && gptl_papi::npapievents () > 0) {
    if (create_and_start_events (0) < 0)
      return GPTLerror ("GPTL: Unthreaded %s: error from GPTLcreate_and_start_events for thread %0\n",
                        thisfunc);

    GPTLthreadid = 0;
  }
#endif

  nthreads = 1;
  return 0;
}

static void print_threadmapping (FILE *fp)
{
  fprintf (fp, "\n");
  fprintf (fp, "GPTLthreadid[0] = 0\n");
}

