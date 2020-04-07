#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include "private.h"

namespace {
  using namespace gptl_private;
  typedef struct {
    int max_depth;
    int max_namelen;
    int max_chars2pr;
  } Outputfmt;

  extern "C" {
    int get_longest_omp_namelen (void);
    void print_titles (int, FILE *, Outputfmt *);
    int construct_tree (Timer *);
    char *methodstr (GPTL_Method);
    int newchild (Timer *, Timer *);
    int get_outputfmt (const Timer *, const int, const int, Outputfmt *);
    void fill_output (int, int, int, Outputfmt *);
    int get_max_namelen (Timer *);
    int is_descendant (const Timer *, const Timer *);
    int is_onlist (const Timer *, const Timer *);
    void printstats (const Timer *, FILE *, int, int, bool, double, double, const Outputfmt);
    void print_multparentinfo (FILE *, Timer *);
    void add (Timer *, const Timer *);
    void printself_andchildren (const Timer *, FILE *, int, int, double, double, Outputfmt);
    void print_hashstats (FILE *);
    float meanhashvalue (Hashentry *, int);
    void print_memstats (FILE *, Timer **, int, int);
    void translate_truncated_names (int, FILE *);
    int rename_duplicate_addresses (void);
  }
}
#endif
