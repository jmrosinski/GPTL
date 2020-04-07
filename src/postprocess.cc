#include "config.h"    // must be first include
#include "private.h"
#include "once.h"
#include "getoverhead.h"
#include "postprocess.h"
#include "util.h"
#include "thread.h"

#include <stdio.h>
#include <stdlib.h>

static const int indent_chars = 2;       // Number of chars to indent

// Public functions for output

/* 
** GPTLpr: Print values of all timers
**
** Input arguments:
**   id: integer to append to string "timing."
**
** Return value: 0 (success) or gptl_util::error (failure)
*/
int GPTLpr (const int id) // output file will be named "timing.<id>" or stderr if negative or huge
{
  using namespace gptl_util;
  char outfile[14];       // name of output file: timing.xxxxxx
  static const char *thisfunc = "GPTLpr";

  // Not great hack to force output to stderr: input a negative or huge number
  if (id < 0 || id > 999999) {
    gptl_util::note ("%s id=%d means output will be written to stderr\n", thisfunc, id);
    sprintf (outfile, "stderr");
  } else {
    sprintf (outfile, "timing.%6.6d", id);
  }

  if (GPTLpr_file (outfile) != 0)
    return gptl_util::error ("%s: Error in GPTLpr_file\n", thisfunc);

  return 0;
}

/* 
** GPTLpr_file: Print values of all timers
**
** Input arguments:
**   outfile: Name of output file to write
**
** Return value: 0 (success) or gptl_util::error (failure)
*/
int GPTLpr_file (const char *outfile)
{
  using namespace gptl_util;
  using namespace gptl_once;
  using namespace gptl_thread;
  FILE *fp;                 // file handle to write to
  Timer *ptr;               // walk through master thread linked list
  Timer *tptr;              // walk through slave threads linked lists
  Timer sumstats;           // sum of same timer stats over threads
  Outputfmt outputfmt;      // max depth, namelen, chars2pr
  int n, t;                 // indices
  int ndup;                 // number of duplicate auto-instrumented addresses
  unsigned long totcount;   // total timer invocations
  float *sum;               // sum of overhead values (per thread)
  float osum;               // sum of overhead over threads
  bool found;               // matching name found: exit linked list traversal
  bool foundany;            // multiple threads with matching name => print across-threads results
  bool first;               // flag 1st time entry found
  double self_ohd;          // estimated library overhead in self timer
  double parent_ohd;        // estimated library overhead due to self in parent timer
  float procsiz, rss;       // returned from GPTLget_procsiz
  static const char *thisfunc = "GPTLpr_file";

  if ( ! initialized)
    return gptl_util::error ("%s: GPTLinitialize() has not been called\n", thisfunc);

  // Not great hack to force output to stderr: "output" is the string "stderr"
  if (STRMATCH (outfile, "stderr") || ! (fp = fopen (outfile, "w")))
    fp = stderr;

  // Rename auto-instrumented entries with same name but different address due to lopping
  if ((ndup = rename_duplicate_addresses ()) > 0) {
    fprintf (fp, "%d duplicate auto-instrumented addresses were found and @<num> added to name",
	     ndup);
    if (ndup > 255) {
      fprintf (fp, "%d did NOT have @<num> added because there were too many\n", ndup);
      fprintf (fp, "Consider increasing the value of MAX_CHARS in private.h next run\n");
    }
  }

  // Print a warning if gptl_util::error() was ever called
  if (num_errors > 0) {
    fprintf (fp, "WARNING: gptl_util::error was called at least once during the run.\n");
    fprintf (fp, "Please examine your output for error messages beginning with GPTL...\n");
  }

  // Print a warning if imperfect nesting was encountered
  if (imperfect_nest) {
    fprintf (fp, "WARNING: SOME TIMER CALLS WERE DETECTED TO HAVE IMPERFECT NESTING.\n");
    fprintf (fp, "TIMING RESULTS WILL BE PRINTED WITHOUT INDENTING AND NO PARENT-CHILD\n");
    fprintf (fp, "INDENTING WILL BE DONE.\n");
    fprintf (fp, "ALSO: NO MULTIPLE PARENT INFORMATION WILL BE PRINTED SINCE IT MAY CONTAIN ERRORS\n");
  }

  // A set of nasty ifdefs to tell important aspects of how GPTL was built
#ifdef HAVE_NANOTIME
  if (funclist[funcidx].option == GPTLnanotime) {
    fprintf (fp, "Clock rate = %f MHz\n", cpumhz);
    fprintf (fp, "Source of clock rate was %s\n", clock_source);
    if (strcmp (clock_source, "/proc/cpuinfo") == 0) {
      fprintf (fp, "WARNING: The contents of /proc/cpuinfo can change in variable frequency CPUs");
      fprintf (fp, "Therefore the use of nanotime (register read) is not recommended on machines"
	       "so equipped");
    }
#ifdef BIT64
    fprintf (fp, "  BIT64 was true\n");
#else
    fprintf (fp, "  BIT64 was false\n");
#endif
  }
#endif

#if ( defined THREADED_OMP )
  fprintf (fp, "GPTL was built with THREADED_OMP\n");
#elif ( defined THREADED_PTHREADS )
  fprintf (fp, "GPTL was built with THREADED_PTHREADS\n");
#else
  fprintf (fp, "GPTL was built without threading\n");
#endif

#ifdef HAVE_LIBMPI
  fprintf (fp, "HAVE_LIBMPI was true\n");

#ifdef ENABLE_PMPI
  fprintf (fp, "  ENABLE_PMPI was true\n");
#else
  fprintf (fp, "  ENABLE_PMPI was false\n");
#endif

#else
  fprintf (fp, "HAVE_LIBMPI was false\n");
#endif

#ifdef HAVE_PAPI
  fprintf (fp, "HAVE_PAPI was true\n");
  if (dousepapi) {
    if (gptl_papi::is_multiplexed)
      fprintf (fp, "  PAPI event multiplexing was ON\n");
    else
      fprintf (fp, "  PAPI event multiplexing was OFF\n");
    PAPIprintenabled (fp);
  }
#else
  fprintf (fp, "HAVE_PAPI was false\n");
#endif

#ifdef ENABLE_NESTEDOMP
  fprintf (fp, "ENABLE_NESTEDOMP was true\n");
#else
  fprintf (fp, "ENABLE_NESTEDOMP was false\n");
#endif

#ifdef HAVE_LIBUNWIND
  fprintf (fp, "Autoprofiling capability was enabled with libunwind\n");
#elif defined HAVE_BACKTRACE	  
  fprintf (fp, "Autoprofiling capability was enabled with backtrace\n");
#else
  fprintf (fp, "Autoprofiling capability was NOT enabled\n");
#endif	   

  fprintf (fp, "Underlying timing routine was %s.\n", funclist[funcidx].name);
  (void) gptl_overhead::get_overhead (fp, &self_ohd, &parent_ohd);
  if (dopr_preamble) {
    fprintf (fp, "\nIf overhead stats are printed, they are the columns labeled self_OH and parent_OH\n"
	     "self_OH is estimated as 2X the Fortran layer cost (start+stop) plust the cost of \n"
	     "a single call to the underlying timing routine.\n"
	     "parent_OH is the overhead for the named timer which is subsumed into its parent.\n"
	     "It is estimated as the cost of a single GPTLstart()/GPTLstop() pair.\n"
             "Print method was %s.\n", methodstr (method));
#ifdef ENABLE_PMPI
    fprintf (fp, "\nIf a AVG_MPI_BYTES field is present, it is an estimate of the per-call\n"
             "average number of bytes handled by that process.\n"
             "If timers beginning with sync_ are present, it means MPI synchronization "
             "was turned on.\n");
#endif
    fprintf (fp, "\nIf a \'%%_of\' field is present, it is w.r.t. the first timer for thread 0.\n"
             "If a \'e6_per_sec\' field is present, it is in millions of PAPI counts per sec.\n\n"
             "A '*' in column 1 below means the timer had multiple parents, though the values\n"
             "printed are for all calls. Multiple parent stats appear later in the file in the\n"
	     "section titled \'Multiple parent info\'\n"
	     "A \'!\' in column 1 means the timer is currently ON and the printed timings are only\n"
	     "valid as of the previous GPTLstop. \'!\' overrides \'*\' if the region had multiple\n"
	     "parents and was currently ON.\n\n");
  }

  // Print the process size at time of call to GPTLpr_file
  (void) GPTLget_procsiz (&procsiz, &rss);
  fprintf (fp, "Process size=%f MB rss=%f MB\n\n", procsiz, rss);

  sum = (float *) gptl_util::allocate (nthreads * sizeof (float), thisfunc);
  
  for (t = 0; t < nthreads; ++t) {
    print_titles (t, fp, &outputfmt);
    /*
    ** Print timing stats. If imperfect nesting was detected, print stats by going through
    ** the linked list and do not indent anything due to the possibility of error.
    ** Otherwise, print call tree and properly indented stats via recursive routine. "-1" 
    ** is flag to avoid printing dummy outermost timer, and initialize the depth.
    */
    if (imperfect_nest) {
      for (ptr = timers[t]->next; ptr; ptr = ptr->next) {
	printstats (ptr, fp, t, 0, false, self_ohd, parent_ohd, outputfmt);
      }
    } else {
      printself_andchildren (timers[t], fp, t, -1, self_ohd, parent_ohd, outputfmt);
    }

    // Sum of self+parent overhead across timers is an estimate of total overhead.
    sum[t]   = 0;
    totcount = 0;
    for (ptr = timers[t]->next; ptr; ptr = ptr->next) {
      sum[t]   += ptr->count * (parent_ohd + self_ohd);
      totcount += ptr->count;
    }
    if (wallstats.enabled && overheadstats.enabled)
      fprintf (fp, "\n");
    fprintf (fp, "Overhead sum = %9.3g wallclock seconds\n", sum[t]);
    if (totcount < PRTHRESH)
      fprintf (fp, "Total calls  = %lu\n", totcount);
    else
      fprintf (fp, "Total calls  = %9.3e\n", (float) totcount);
  }

  // Print per-name stats for all threads
  if (dopr_threadsort && nthreads > 1) {
    int nblankchars;
    fprintf (fp, "\nSame stats sorted by timer for threaded regions:\n");
    fprintf (fp, "Thd ");

    // Reset outputfmt contents for multi-threads
    outputfmt.max_depth    = 0;
    outputfmt.max_namelen  = get_longest_omp_namelen ();
    outputfmt.max_chars2pr = outputfmt.max_namelen;
    nblankchars            = outputfmt.max_chars2pr + 1; // + 1 ensures a blank after name
    for (n = 0; n < nblankchars; ++n)  // length of longest multithreaded timer name
      fprintf (fp, " ");
    fprintf (fp, "  Called  Recurse");
    if (cpustats.enabled)
      fprintf (fp, "%9s", cpustats.str);
    if (wallstats.enabled) {
      fprintf (fp, "%9s", wallstats.str);
      if (percent && timers[0]->next)
        fprintf (fp, " %%_of_%4.4s ", timers[0]->next->name);
      if (overheadstats.enabled)
        fprintf (fp, "%9s", overheadstats.str);
    }
#ifdef HAVE_PAPI
    gptl_papi::PAPIprstr (fp);
#endif
    fprintf (fp, "\n");
    // Start at next to skip GPTL_ROOT
    for (ptr = timers[0]->next; ptr; ptr = ptr->next) {      
      // To print sum stats, first create a new timer then copy thread 0
      // stats into it. then sum using "add", and finally print.
      foundany = false;
      first = true;
      sumstats = *ptr;
      for (t = 1; t < nthreads; ++t) {
        found = false;
        for (tptr = timers[t]->next; tptr && ! found; tptr = tptr->next) {
          if (STRMATCH (ptr->name, tptr->name)) {
            // Only print thread 0 when this timer found for other threads
            if (first) {
              first = false;
              fprintf (fp, "%3.3d ", 0);
              printstats (ptr, fp, 0, 0, false, self_ohd, parent_ohd, outputfmt);
            }
            found = true;
            foundany = true;
            fprintf (fp, "%3.3d ", t);
            printstats (tptr, fp, 0, 0, false, self_ohd, parent_ohd, outputfmt);
            add (&sumstats, tptr);
          }
        }
      }
      if (foundany) {
        fprintf (fp, "SUM ");
        printstats (&sumstats, fp, 0, 0, false, self_ohd, parent_ohd, outputfmt);
        fprintf (fp, "\n");
      }
    }
    // Repeat overhead print in loop over threads
    if (wallstats.enabled && overheadstats.enabled) {
      osum = 0.;
      for (t = 0; t < nthreads; ++t) {
        fprintf (fp, "OVERHEAD.%3.3d (wallclock seconds) = %9.3g\n", t, sum[t]);
        osum += sum[t];
      }
      fprintf (fp, "OVERHEAD.SUM (wallclock seconds) = %9.3g\n", osum);
    }
  }
  // For auto-instrumented apps, translate names which have been truncated for output formatting
  for (t = 0; t < nthreads; ++t)
    translate_truncated_names (t, fp);
  
  // Print info about timers with multiple parents ONLY if imperfect nesting was not discovered
  if (dopr_multparent && ! imperfect_nest) {
    for (t = 0; t < nthreads; ++t) {
      bool some_multparents = false;   // thread has entries with multiple parents?
      for (ptr = timers[t]->next; ptr; ptr = ptr->next) {
        if (ptr->nparent > 1) {
          some_multparents = true;
          break;
        }
      }
      if (some_multparents) {
        fprintf (fp, "\nMultiple parent info for thread %d:\n", t);
        if (dopr_preamble && t == 0) {
          fprintf (fp, "Columns are count and name for the listed child\n"
                   "Rows are each parent, with their common child being the last entry, "
                   "which is indented.\n"
                   "Count next to each parent is the number of times it called the child.\n"
                   "Count next to child is total number of times it was called by the "
                   "listed parents.\n\n");
        }
        for (ptr = timers[t]->next; ptr; ptr = ptr->next)
          if (ptr->nparent > 1)
            print_multparentinfo (fp, ptr);
      }
    }
  }
  // Print hash table stats
  if (dopr_collision)
    print_hashstats (fp);
  // Print stats on GPTL memory usage
  print_memstats (fp, timers, tablesize, maxthreads);

  free (sum);

  if (fp != stderr && fclose (fp) != 0)
    fprintf (stderr, "%s: Attempt to close %s failed\n", thisfunc, outfile);

  return 0;
}

// Functions local to this file: put in anonymous namespace
namespace {
  extern "C" {
    // get_longest_omp_namelen: Discover longest name shared across threads
    int get_longest_omp_namelen (void)
    {
      Timer *ptr;
      Timer *tptr;
      bool found;
      int longest = 0;

      for (ptr = timers[0]->next; ptr; ptr = ptr->next) {
	for (int t = 1; t < gptl_thread::nthreads; ++t) {
	  found = false;
	  for (tptr = timers[t]->next; tptr && ! found; tptr = tptr->next) {
	    if (STRMATCH (tptr->name, ptr->name)) {
	      found = true;
	      longest = MAX (longest, strlen (ptr->name));
	    }
	    if (found) // Found matching name: done with this thread
	      break;
	  }
	  if (found)   // No need to check other threads for the same name
	    break;
	}
      }
      return longest;
    }

    /* 
    ** print_titles: Print headings to output file. If imperfect nesting was detected, print simply
    **               by following the linked list. Otherwise, indent using parent-child 
    **               relationships.
    **
    ** Input arguments:
    **   t:         thread number
    **   fp:        file pointer to write to
    **   outputfmt: max depth, namelen, chars2pr
    */
    void print_titles (int t, FILE *fp, Outputfmt *outputfmt)
    {
      int n;
      int ret;
      int nblankchars;
      static const char *thisfunc = "print_titles";

      /*
      ** Construct tree for printing timers in parent/child form. get_outputfmt() must be called 
      ** AFTER construct_tree() because it relies on the per-parent children arrays being complete.
      ** Initialize outputfmt, then call recursive routine to fill its contents
      */
      outputfmt->max_depth    = 0;
      outputfmt->max_namelen  = 0;
      outputfmt->max_chars2pr = 0;

      if (imperfect_nest) {
	outputfmt->max_namelen  = get_max_namelen (timers[t]);
	outputfmt->max_chars2pr = outputfmt->max_namelen;
	nblankchars             = outputfmt->max_namelen + 1;
      } else {
	if (construct_tree (timers[t]) != 0)
	  printf ("GPTL: %s: failure from construct_tree: output will be incomplete\n", thisfunc);
	
	// Start at GPTL_ROOT because that is the parent of all timers => guarantee traverse
	// full tree. -1 is initial call tree depth
	ret = get_outputfmt (timers[t], -1, indent_chars, outputfmt);
#ifdef DEBUG
	printf ("%s t=%d got outputfmt=%d %d %d\n",
		thisfunc, t, outputfmt->max_depth, outputfmt->max_namelen, outputfmt->max_chars2pr);
#endif
	nblankchars = indent_chars + outputfmt->max_chars2pr + 1;
      }

      if (t > 0)
	fprintf (fp, "\n");
      fprintf (fp, "Stats for thread %d:\n", t);

      // Start title printing after longest (indent_chars + name) + 1
      for (n = 0; n < nblankchars; ++n)
	fprintf (fp, " ");
      fprintf (fp, "  Called  Recurse");

      // Print strings for enabled timer types
      if (cpustats.enabled)
	fprintf (fp, "%9s", cpustats.str);
      if (wallstats.enabled) {
	fprintf (fp, "%9s", wallstats.str);
	if (gptl_once::percent && timers[0]->next)
	  fprintf (fp, " %%_of_%4.4s", timers[0]->next->name);
	if (overheadstats.enabled)
	  fprintf (fp, "%9s", overheadstats.str);
      }
#ifdef ENABLE_PMPI
      fprintf (fp, " AVG_MPI_BYTES");
#endif

#ifdef HAVE_PAPI
      PAPIprstr (fp);
#endif
      fprintf (fp, "\n");
      return;
    }

    /* 
    ** construct_tree: Build the parent->children tree starting with knowledge of
    **                 parent list for each child.
    **
    ** Input arguments:
    **   method:  method to be used to define the links
    **
    ** Input/Output arguments:
    **   timerst: Linked list of timers. "children" array for each timer will be constructed
    **
    ** Return value: 0 (success) or gptl_util::error (failure)
    */
    int construct_tree (Timer *timerst)
    {
      Timer *ptr;       // traverse linked list
      Timer *pptr = 0;  // parent (init to NULL to avoid compiler warning)
      int nparent;      // number of parents
      int maxcount;     // max calls by a single parent
      int n;            // loop over nparent
      static const char *thisfunc = "construct_tree";

      // Walk the linked list to build the parent-child tree, using whichever
      // mechanism is in place. newchild() will prevent loops.
      for (ptr = timerst; ptr; ptr = ptr->next) {
	switch (gptl_once::method) {
	case GPTLfirst_parent:
	  if (ptr->nparent > 0) {
	    pptr = ptr->parent[0];
	    if (newchild (pptr, ptr) != 0) {};
	  }
	  break;
	case GPTLlast_parent:
	  if (ptr->nparent > 0) {
	    nparent = ptr->nparent;
	    pptr = ptr->parent[nparent-1];
	    if (newchild (pptr, ptr) != 0) {};
	  }
	  break;
	case GPTLmost_frequent:
	  maxcount = 0;
	  for (n = 0; n < ptr->nparent; ++n) {
	    if (ptr->parent_count[n] > maxcount) {
	      pptr = ptr->parent[n];
	      maxcount = ptr->parent_count[n];
	    }
	  }
	  if (maxcount > 0) {   // not an orphan
	    if (newchild (pptr, ptr) != 0) {};
	  }
	  break;
	case GPTLfull_tree:
	  for (n = 0; n < ptr->nparent; ++n) {
	    pptr = ptr->parent[n];
	    if (newchild (pptr, ptr) != 0) {};
	  }
	  break;
	default:
	  return gptl_util::error ("GPTL: %s: method %d is not known\n",
				   thisfunc, gptl_once::method);
	}
      }
      return 0;
    }

    // methodstr: Return a pointer to a string which represents the print method to be used
    char *methodstr (GPTL_Method method)
    {
      static char first_parent[]  = "first_parent";
      static char last_parent[]   = "last_parent";
      static char most_frequent[] = "most_frequent";
      static char full_tree[]     = "full_tree";

      if (method == GPTLfirst_parent)
	return first_parent;
      else if (method == GPTLlast_parent)
	return last_parent;
      else if (method == GPTLmost_frequent)
	return most_frequent;
      else if (method == GPTLfull_tree)
	return full_tree;
      else
	return unknown;
    }

    /* 
    ** newchild: Add an entry to the children list of parent. Use function
    **   is_descendant() to prevent infinite loops. 
    **
    ** Input arguments:
    **   child:  child to be added
    **
    ** Input/output arguments:
    **   parent: parent node which will have "child" added to its "children" array
    **
    ** Return value: 0 (success) or gptl_util::error (failure)
    */
    int newchild (Timer *parent, Timer *child)
    {
      int nchildren;     // number of children (temporary)
      Timer **chptr;     // array of pointers to children
      static const char *thisfunc = "newchild";

      if (parent == child)
	return gptl_util::error ("%s: child %s can't be a parent of itself\n", thisfunc, child->name);

      // To guarantee no loops, ensure that proposed parent isn't already a descendant of 
      // proposed child
      if (is_descendant (child, parent)) {
	return gptl_util::error ("GPTL: %s: loop detected: NOT adding %s to descendant list of %s. "
			  "Proposed parent is in child's descendant path.\n",
			  thisfunc, child->name, parent->name);
      }

      // Add child to parent's array of children if it isn't already there (e.g. by an earlier call
      // to GPTLpr*)
      if ( ! is_onlist (child, parent)) {
	++parent->nchildren;
	nchildren = parent->nchildren;
	chptr = (Timer **) realloc (parent->children, nchildren * sizeof (Timer *));
	if ( ! chptr)
	  return gptl_util::error ("%s: realloc error\n", thisfunc);
	parent->children = chptr;
	parent->children[nchildren-1] = child;
      }
      return 0;
    }

    /* 
    ** get_outputfmt: Determine max depth, name length, and chars to be printed prior to data by 
    **   traversing tree recursively
    **
    ** Input arguments:
    **   ptr:    Starting timer
    **   depth:  current depth when function invoked 
    **   indent: number of chars to indent
    **
    ** Output arguments:
    **   outputfmt: output format
    **
    ** Return value: maximum characters to be printed prior to data
    */
    int get_outputfmt (const Timer *ptr, const int depth, const int indent, Outputfmt *outputfmt)
    {
      int ret;
      int namelen  = strlen (ptr->name);
      int chars2pr = namelen + indent*depth;

      if (ptr->nchildren == 0) {
	fill_output (depth, namelen, chars2pr, outputfmt);
	return 0;
      }

      for (int n = 0; n < ptr->nchildren; ++n) {
	ret = get_outputfmt (ptr->children[n], depth+1, indent, outputfmt);
	fill_output (depth, namelen, chars2pr, outputfmt);
      }
      return 0;
    }

    void fill_output (int depth, int namelen, int chars2pr, Outputfmt *outputfmt)
    {
      if (depth > outputfmt->max_depth)
	outputfmt->max_depth = depth;

      if (namelen > outputfmt->max_namelen)
	outputfmt->max_namelen = namelen;
      
      if (chars2pr > outputfmt->max_chars2pr)
	outputfmt->max_chars2pr = chars2pr;
    }

    /*
    ** get_max_namelen: Discover maximum name length. Called only when imperfect_nest is true
    ** Input arguments:
    **   ptr: Starting timer
    **
    ** Return value:
    **   max name length
    */
    int get_max_namelen (Timer *timers)
    {
      Timer *ptr;            // traverse linked list
      int namelen;           // name length for individual timer
      int max_namelen = 0;   // return value

      for (ptr = timers; ptr; ptr = ptr->next) {
	namelen = strlen (ptr->name);
	if (namelen > max_namelen)
	  max_namelen = namelen;
      }
      return max_namelen;
    }

    /* 
    ** is_descendant: Determine whether node2 is in the descendant list for node1
    **
    ** Input arguments:
    **   node1: starting node for recursive search
    **   node2: node to be searched for
    **
    ** Return value: true or false
    */
    int is_descendant (const Timer *node1, const Timer *node2)
    {
      // Breadth before depth for efficiency
      for (int n = 0; n < node1->nchildren; ++n)
	if (node1->children[n] == node2)
	  return 1;
      
      for (int n = 0; n < node1->nchildren; ++n)
	if (is_descendant (node1->children[n], node2))
	  return 1;
      
      return 0;
    }

    /* 
    ** is_onlist: Determine whether child is in parent's list of children
    **
    ** Input arguments:
    **   child: who to search for
    **   parent: search through his list of children
    **
    ** Return value: true or false
    */
    int is_onlist (const Timer *child, const Timer *parent)
    {
      for (int n = 0; n < parent->nchildren; ++n) {
	if (child == parent->children[n])
	  return 1;
      }
      return 0;
    }

    /* 
    ** printstats: print a single timer
    **
    ** Input arguments:
    **   timer:        timer for which to print stats
    **   fp:           file descriptor to write to
    **   t:            thread number
    **   depth:        depth to indent timer
    **   doindent:     whether indenting will be done
    **   tot_overhead: underlying timing routine overhead
    */
    void printstats (const Timer *timer, FILE *fp, int t, int depth, bool doindent,
		     double self_ohd, double parent_ohd, const Outputfmt outputfmt)
    {
      int indent;          // index for indenting
      int extraspace;      // for padding to length of longest name
      float fusr;          // user time as float
      float fsys;          // system time as float
      float usrsys;        // usr + sys
      float elapse;        // elapsed time
      float wallmax;       // max wall time
      float wallmin;       // min wall time
      float ratio;         // percentage calc
      static const char *thisfunc = "printstats";
      
      if (timer->onflg && gptl_once::verbose)
	fprintf (stderr, "GPTL: %s: timer %s had not been turned off\n", thisfunc, timer->name);

      // Flag regions having multiple parents with a "*" in column 1
      // TODO: The following ASSUMES indent_chars=2!!!
      if (doindent) {
	if (timer->onflg)
	  fprintf (fp, "! ");
	else if (timer->nparent > 1)
	  fprintf (fp, "* ");
	else
	  fprintf (fp, "  ");

	// Indent to depth of this timer
	for (indent = 0; indent < depth*indent_chars; ++indent)
	  fprintf (fp, " ");
      }

      fprintf (fp, "%s", timer->name);

      // Pad to most chars to print
      extraspace = outputfmt.max_chars2pr - (depth*indent_chars + strlen (timer->name));
      for (int i = 0; i < extraspace; ++i)
	fprintf (fp, " ");

      if (timer->count < PRTHRESH) {
	if (timer->nrecurse > 0)
	  fprintf (fp, " %8lu %8lu", timer->count, timer->nrecurse);
	else
	  fprintf (fp, " %8lu     -   ", timer->count);
      } else {
	if (timer->nrecurse > 0)
	  fprintf (fp, " %8.1e %8.0e",    (float) timer->count, (float) timer->nrecurse);
	else
	  fprintf (fp, " %8.1e     -   ", (float) timer->count);
      }
      
      if (cpustats.enabled) {
	fusr = timer->cpu.accum_utime / (float) gptl_once::ticks_per_sec;
	fsys = timer->cpu.accum_stime / (float) gptl_once::ticks_per_sec;
	usrsys = fusr + fsys;
	fprintf (fp, " %8.1e %8.1e %8.1e", fusr, fsys, usrsys);
      }

      if (wallstats.enabled) {
	elapse = timer->wall.accum;
	wallmax = timer->wall.max;
	wallmin = timer->wall.min;

	if (elapse < 0.01)
	  fprintf (fp, " %8.2e", elapse);
	else
	  fprintf (fp, " %8.3f", elapse);
	
	if (wallmax < 0.01)
	  fprintf (fp, " %8.2e", wallmax);
	else
	  fprintf (fp, " %8.3f", wallmax);

	if (wallmin < 0.01)
	  fprintf (fp, " %8.2e", wallmin);
	else
	  fprintf (fp, " %8.3f", wallmin);
	
	if (gptl_once::percent && timers[0]->next) {
	  ratio = 0.;
	  if (timers[0]->next->wall.accum > 0.)
	    ratio = (timer->wall.accum * 100.) / timers[0]->next->wall.accum;
	  fprintf (fp, " %8.2f", ratio);
	}
	
	if (overheadstats.enabled) {
	  fprintf (fp, " %8.3f %8.3f", timer->count*self_ohd, timer->count*parent_ohd);
	}
      }

#ifdef ENABLE_PMPI
      if (timer->nbytes == 0.)
	fprintf (fp, "       -      ");
      else
	fprintf (fp, "%13.3e ", timer->nbytes / timer->count);
#endif
  
#ifdef HAVE_PAPI
      PAPIpr (fp, &timer->aux, t, timer->count, timer->wall.accum);
#endif

      fprintf (fp, "\n");
    }

    // print_multparentinfo: print info about regions with multiple parents
    void print_multparentinfo (FILE *fp, Timer *ptr)
    {
      if (ptr->norphan > 0) {
	if (ptr->norphan < PRTHRESH)
	  fprintf (fp, "%8u %-32s\n", ptr->norphan, "ORPHAN");
	else
	  fprintf (fp, "%8.1e %-32s\n", (float) ptr->norphan, "ORPHAN");
      }

      for (int n = 0; n < ptr->nparent; ++n) {
	char *parentname;
	if (ptr->parent[n]->longname)
	  parentname = ptr->parent[n]->longname;
	else
	  parentname = ptr->parent[n]->name;
	
	if (ptr->parent_count[n] < PRTHRESH)
	  fprintf (fp, "%8d %-s\n", ptr->parent_count[n], parentname);
	else
	  fprintf (fp, "%8.1e %-s\n", (float) ptr->parent_count[n], parentname);
      }

      if (ptr->count < PRTHRESH)
	if (ptr->longname)
	  fprintf (fp, "%8lu   %-s\n\n", ptr->count, ptr->longname);
	else
	  fprintf (fp, "%8lu   %-s\n\n", ptr->count, ptr->name);
      else
	if (ptr->longname)
	  fprintf (fp, "%8.1e   %-s\n\n", (float) ptr->count, ptr->longname);
	else
	  fprintf (fp, "%8.1e   %-s\n\n", (float) ptr->count, ptr->name);
    }

    // add: add the contents of timer tin to timer tout
    void add (Timer *tout, const Timer *tin)
    {
      tout->count += tin->count;

      if (wallstats.enabled) {
	tout->wall.accum += tin->wall.accum;
	tout->wall.max = MAX (tout->wall.max, tin->wall.max);
	tout->wall.min = MIN (tout->wall.min, tin->wall.min);
      }

      if (cpustats.enabled) {
	tout->cpu.accum_utime += tin->cpu.accum_utime;
	tout->cpu.accum_stime += tin->cpu.accum_stime;
      }
#ifdef HAVE_PAPI
      PAPIadd (&tout->aux, &tin->aux);
#endif
    }

    // printself_andchildren: Recurse through call tree, printing stats for self, then children
    void printself_andchildren (const Timer *ptr, FILE *fp, int t, int depth, 
				double self_ohd, double parent_ohd, Outputfmt outputfmt)
    {
      if (depth > -1)     // -1 flag is to avoid printing stats for dummy outer timer
	printstats (ptr, fp, t, depth, true, self_ohd, parent_ohd, outputfmt);

      for (int n = 0; n < ptr->nchildren; n++)
	printself_andchildren (ptr->children[n], fp, t, depth+1, self_ohd, parent_ohd, outputfmt);
    }

    void print_hashstats (FILE *fp)
    {
      int totent;               // per-thread collision count (diagnostic)
      int nument;               // per-index collision count (diagnostic)

      // Diagnostics for collisions and GPTL memory usage
      int num_zero;             // number of buckets with 0 collisions
      int num_one;              // number of buckets with 1 collision
      int num_two;              // number of buckets with 2 collisions
      int num_more;             // number of buckets with more than 2 collisions
      int most;                 // biggest collision count
      bool first;

      for (int t = 0; t < gptl_thread::nthreads; t++) {
	first = true;
	totent   = 0;
	num_zero = 0;
	num_one  = 0;
	num_two  = 0;
	num_more = 0;
	most     = 0;

	for (int i = 0; i < gptl_private::tablesize; i++) {
	  nument = gptl_private::hashtable[t][i].nument;
	  if (nument > 1) {
	    totent += nument-1;
	    if (first) {
	      first = false;
	      fprintf (fp, "\nthread %d had some hash collisions:\n", t);
	    }
	    fprintf (fp, "hashtable[%d][%d] had %d entries:", t, i, nument);
	    for (int ii = 0; ii < nument; ii++)
	      fprintf (fp, " %s", gptl_private::hashtable[t][i].entries[ii]->name);
	    fprintf (fp, "\n");
	  }
	  switch (nument) {
	  case 0:
	    ++num_zero;
	    break;
	  case 1:
	    ++num_one;
	    break;
	  case 2:
	    ++num_two;
	    break;
	  default:
	    ++num_more;
	    break;
	  }
	  most = MAX (most, nument);
	}
    
	if (totent > 0) {
	  fprintf (fp, "Total collisions thread %d = %d\n", t, totent);
	  fprintf (fp, "Entry information:\n");
	  fprintf (fp, "num_zero = %d num_one = %d num_two = %d num_more = %d\n",
		   num_zero, num_one, num_two, num_more);
	  fprintf (fp, "Most = %d\n", most);
	}
      }
      fprintf (fp, "Size of hash table was %d\n", gptl_private::tablesize);
      fprintf (fp, "Mean hash index for thread 0 was %f\n",
	       meanhashvalue (hashtable[0], tablesize));
    }
  
    float meanhashvalue (Hashentry *hashtable, int tablesize)
    {
      float sum = 0.;  // used to calculate mean
      int nument;
      int totent = 0;  // number of entries
  
      for (int i = 1; i < tablesize; ++i) {
	nument = hashtable[i].nument;
	if (nument > 0) {
	  sum += (float) (nument * i);
	  totent += hashtable[i].nument;
	}
      }
      if (totent == 0)
	return (float) 0.;
      else
	return sum / totent;
    }

    void print_memstats (FILE *fp, Timer **timers, int tablesize, int maxthreads)
    {
      Timer *ptr;               // walk through linked list
      float pchmem = 0.;        // parent/child array memory usage
      float regionmem = 0.;     // timer memory usage
      float papimem = 0.;       // PAPI stats memory usage
      float hashmem;            // hash table memory usage
      float callstackmem;       // callstack memory usage
      float totmem;             // total GPTL memory usage
      int numtimers;            // number of timers

      hashmem = (float) sizeof (Hashentry) * tablesize * maxthreads;  /* fixed size of table */
      callstackmem = (float) sizeof (Timer *) * MAX_STACK * maxthreads;
      for (int t = 0; t < gptl_thread::nthreads; t++) {
	numtimers = 0;
	for (ptr = timers[t]->next; ptr; ptr = ptr->next) {
	  ++numtimers;
	  pchmem  += (float) sizeof (Timer *) * (ptr->nchildren + ptr->nparent);
	}
	hashmem   += (float) numtimers * sizeof (Timer *);
	regionmem += (float) numtimers * sizeof (Timer);
#ifdef HAVE_PAPI
	papimem += (float) numtimers * sizeof (Papistats);
#endif
      }

      totmem = hashmem + regionmem + pchmem + callstackmem;
      fprintf (fp, "\n");
      fprintf (fp, "Total GPTL memory usage = %g KB\n", totmem*.001);
      fprintf (fp, "Components:\n");
      fprintf (fp, "Hashmem                 = %g KB\n" 
               "Regionmem               = %g KB (papimem portion = %g KB)\n"
               "Parent/child arrays     = %g KB\n"
               "Callstackmem            = %g KB\n",
	       hashmem*.001, regionmem*.001, papimem*.001, pchmem*.001, callstackmem*.001);

      gptl_thread::print_threadmapping (fp);
    }

    /* 
    ** translate_truncated_names: For auto-profiled entries, print to output file the translation of
    **                            truncated names to full signature
    **
    ** Input args: t=thread number
    **             fp=file pointer
    */
    void translate_truncated_names (int t, FILE *fp)
    {
      Timer *ptr;

      fprintf (fp, "thread %d long name translations (empty when no auto-instrumentation):\n", t);
      for (ptr = timers[t]->next; ptr; ptr = ptr->next) {
	if (ptr->longname)
	  fprintf (fp, "%s = %s\n", ptr->name, ptr->longname);
      }
    }

    /* 
    ** rename_duplcicate_addresses: Create unique name for auto-profiled entries that are 
    **                              now duplicates due to truncation
    **
    ** Return value: max number of duplicates found
    */
    int rename_duplicate_addresses ()
    {
      Timer *ptr;       // iterate through timers
      Timer *testptr;   // start at ptr, iterate through remaining timers
      int nfound = 0;   // number of duplicates found
      static const char *thisfunc = "rename_duplicate_addresses";

      for (int t = 0; t < gptl_thread::nthreads; ++t) {
	for (ptr = timers[t]; ptr; ptr = ptr->next) {
	  unsigned int idx = 0;
	  // Check only entries with a longname and don't have '@' in their name
	  // The former means auto-instrumented
	  // The latter means the name hasn't yet had the proper '@<number>' appended.
	  if (ptr->longname && ! index (ptr->name, '@')) {
	    for (testptr = ptr->next; testptr; testptr = testptr->next) {
	      if (testptr->longname && STRMATCH (ptr->name, testptr->name)) {
		// Add string "@<idx>" to end of testptr->name to indicate duplicate auto-profiled
		// name for multiple addresses. Probable inlining issue.
		if (++idx < 4096)    // 4095 is the max integer printable in 3 hex chars
		  snprintf (&testptr->name[MAX_CHARS-4], 5, "@%X", idx);
		else
		  snprintf (&testptr->name[MAX_CHARS-4], 5, "@MAX");
	      }
	    }
	    // @<number> has been added to duplicates. Now add @0 to original
	    snprintf (&ptr->name[MAX_CHARS-4], 5, "@%X", 0);
	    nfound = MAX (nfound, idx);
	  }
	}
      }
      return nfound;
    }
  }
}
