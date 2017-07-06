#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>

#define MAXSIZ_LINE 256
#define MAXSIZ_NAME 128

int listmax = 1;          // Default: just list one item

struct Item {
  int nlines_data;        // current line number into data
  char name[MAXSIZ_NAME]; // "name" from gptl output
  float ncalls;           // number of calls (unused--could be int or float)
  int nranks;             // number of ranks (unused)
  float mean_time;        // mean time (unused)
  float std_dev;          // std deviation (unused)
  float wallmax;          // the 1st item we're looking for
  float wallmin;          // the 2nd item we're looking for
};

struct List {
  char name[MAXSIZ_NAME];
  float value[2];
  float diff;
};

void maybe_insert (float, struct List *, struct Item *, int *);
void printlist (char *, char *, struct List *, int);
int scanit (const char *, struct Item *);

int main (int argc, char **argv)
{
  int c;           // for getopt parsing
  int n;           // loop index
  int nitems;      // returned from fscanf
  int ret;         // return code
  char line[MAXSIZ_LINE];  // input line from file
  char **namesav_wall;
  char **namesav_rel;
  float walldiff;           // walltime diff max - min
  float reldiff;            // relative walltime diff max vs. min
  float mindiff = 1.0;      // Default: Throw away all diffs less than 1.0
  FILE *fp = 0;             // file pointers

  struct Item item;
  struct List *difflist;
  struct List *rdifflist;
  int listsize_diff;
  int listsize_rdiff;
  static const char *thisfunc = "main";

  while ((c = getopt (argc, argv, "m:n:")) != -1) {
    switch (c) {
    case 'm':
      mindiff = atof (optarg);
      break;
    case 'n':
      if ((listmax = atoi (optarg)) < 1) {
	printf ("-n listmax must be > 0\n");
	return -1;
      }
      break;
    default:
      printf ("Extraneous argument %s\n", optarg);
    }
  }

  // Exactly 1 argument needed after optional flags
  if (optind >= argc || optind < argc-1) {
    printf ("Usage: %s [-m mindiff] [-n listmax] file\n", argv[0]);
    return -1;
  }

  printf ("Not considering diffs less than %f\n", mindiff);

  item.nlines_data = 0;
  // open each file
  if ( ! (fp = fopen (argv[optind+n], "r"))) {
    printf ("Failure to open input file %s for reading\n", argv[optind+n]);
    return -1;
  }
  // Skip over input until sentinel "name" is reached
  line[0] = ' ';  // Ensure line will not have correct data if fgets fails
  do {
    if ( ! fgets (line, sizeof line, fp)) {
      printf ("Never found 'name' in %s\n", argv[optind+n]);
      return -1;
    }
  } while (strncmp (line, "name ", 5) != 0);
  
  difflist = malloc (listmax * sizeof (struct List));
  rdifflist = malloc (listmax * sizeof (struct List));
  listsize_diff = 0;
  listsize_rdiff = 0;

  // Read through the file, comparing wallmax vs. wallmin entries
  while (fgets (line, sizeof line, fp)) {
    ++item.nlines_data;
    if ((ret = scanit (line, &item)) == 1) {
      printf ("%s: scanit reached EOF on file 0: Printing results:\n", thisfunc);
      break;
    } else if (ret != 9 && ret != 8) {
      printf ("%s: Strange return=%d from scanit: Stopping\n", thisfunc, ret);
      return -1;
    }
    walldiff = fabsf (item.wallmax - item.wallmin);
    if (walldiff > mindiff) {
      maybe_insert (walldiff, difflist, &item, &listsize_diff);
      // Avoid division by zero
      if (item.wallmax != item.wallmin) {
	reldiff = walldiff / (0.5*(item.wallmax + item.wallmin));
	maybe_insert (reldiff, rdifflist, &item, &listsize_rdiff);
      }
    }
  }

  printlist ("Difflist", "diff ", difflist, listsize_diff);
  printlist ("\nRelative difflist", "rdiff", rdifflist, listsize_rdiff);
  return 0;
}
  
void maybe_insert (float diff, struct List *list, struct Item *item, int *listsize)
{
  int n, nn;

#ifdef DEBUG
  printf ("\ntesting diff=%f item=%s vals=%f %f\n", diff, item->name, item->wallmax, item->wallmin);
#endif
  for (n = 0; n < *listsize; ++n) {
    if (diff > list[n].diff) {
      // Move the list down to make room for the new (sorted) entry
      // Note if adding to end of list (n=*listsize-1) no movement will be done
#ifdef DEBUG
      printf ("Freeing up list at n=%d name=%s by shifting thru n=%d name=%s\n",
	      n, list[n].name, *listsize-1, list[*listsize-1].name);
#endif
      for (nn = *listsize-1; nn > n-1; --nn) {
	if (nn < listmax-1) {
	  strcpy (list[nn+1].name, list[nn].name);
	  list[nn+1].diff        = list[nn].diff;
	  list[nn+1].value[0]    = list[nn].value[0];
	  list[nn+1].value[1]    = list[nn].value[1];
	}
      }
      break;
    }
  }

  if (n < listmax) {
    // Insert the new value at position n (note n can be 0 if list is empty)
    strcpy (list[n].name, item[0].name);
    list[n].diff     = diff;
    list[n].value[0] = item->wallmax;
    list[n].value[1] = item->wallmin;
#ifdef DEBUG
    printf ("Inserting name=%s into position %d diff=%f values= %f %f\n",
	    list[n].name, n, list[n].diff, list[n].value[0], list[n].value[1]);
#endif
    // Increment listsize only if new size wouldn't exceed max
    // Note if list is full we're replacing the smallest item
    if (*listsize < listmax) {
      ++*listsize;
#ifdef DEBUG
      printf ("New listsize=%d\n", *listsize);
#endif
    }
#ifdef DEBUG
    printlist ("End of maybe_insert", "diff ", list, *listsize);
#endif
  }
  return;
}

void printlist (char *str, char *difftype, struct List *list, int listsize)
{
  int n;

  printf ("%s\n", str);
  for (n = 0; n < listsize; ++n) {
    printf ("n=%d %s=%9.3f values=%9.3f %9.3f name=%s\n",
	    n, difftype, list[n].diff, list[n].value[0], list[n].value[1], list[n].name);
  }
}

int scanit (const char *line, struct Item *item)
{
  int rank;
  int thread;
  int nitems;

  // 1st check for threaded case
  nitems = sscanf (line, "%s %f %d %f %f %f (%d %d) %f",
		   item->name, &(item->ncalls), &(item->nranks), &(item->mean_time),
		   &(item->std_dev), &(item->wallmax), &rank, &thread, &(item->wallmin));

  // next check for unthreaded case
  if (nitems != 9) {
    nitems = sscanf (line, "%s %f %d %f %f %f (%d) %f",
		     item->name, &(item->ncalls), &(item->nranks), &(item->mean_time),
		     &(item->std_dev), &(item->wallmax), &rank, &(item->wallmin));
  }

  if (nitems != 8 && nitems != 9) {
    printf ("scanit: sscanf returned %d items expected 9 or 8\n", nitems);
    if (nitems > 0)
      printf ("item 0 got %s\n", item->name);
    if (nitems > 1)
      printf ("item 1 got %d\n", item->ncalls);
  }
  return nitems;
}
