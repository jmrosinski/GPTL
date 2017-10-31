#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>

#define MAXSIZ_LINE 256
#define MAXSIZ_NAME 128

int listmax = 1;          // Default: just list one item

struct Item {
  int nlines_hdr;         // number of lines of header stuff
  int nlines_data;        // current line number into data
  char name[MAXSIZ_NAME]; // "name" from gptl output
  float ncalls;           // number of calls (unused--could be int or float)
  int nranks;             // number of ranks (unused)
  float mean_time;        // mean time (unused)
  float std_dev;          // std deviation (unused)
  float wallmax;          // the item we're looking for
};

struct List {
  char name[MAXSIZ_NAME];
  float value[2];
  float diff;
};

int get_matching_line (const char *, FILE *, struct Item *);    // find line in file2 matching file1
void maybe_insert (float, struct List *, struct Item *, int *); // insert an item into list
void printlist (char *, char *, struct List *, int);            // print results
int scanit (const char *, struct Item *);       // parse line contents into components
int skipjunk (FILE *, int, char *);             // skips the lines up to "name ..."

int main (int argc, char **argv)
{
  int c;           // for getopt parsing
  int n;           // loop index
  int nitems;      // returned from fscanf
  int ret;         // return code
  char line[MAXSIZ_LINE]; // input line from file
  char name0[MAXSIZ_NAME], name1[MAXSIZ_NAME]; // entry name
  char **namesav_wall;
  char **namesav_rel;
  float walldiff;           // walltime diff file1 - file2
  float reldiff;            // relative walltime diff file1 - file2
  float mindiff = 1.0;      // Default: Throw away all diffs less than 1.0
  FILE *fp[2] = {0,0};      // file pointers

  struct Item item[2];
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

  // Exactly 2 arguments needed after optional flags
  if (optind >= argc || optind < argc-2) {
    printf ("Usage: %s [-m mindiff] [-n listmax] file1 file2\n", argv[0]);
    return -1;
  }

  printf ("Not considering diffs less than %f\n", mindiff);

  for (n = 0; n < 2; ++n) {
    item[n].nlines_hdr = 0;
    item[n].nlines_data = 0;
    // open each file
    if ( ! (fp[n] = fopen (argv[optind+n], "r"))) {
      printf ("Failure to open input file %s for reading\n", argv[optind+n]);
      return -1;
    }
    // Skip over input until sentinel "name" is reached
    line[0] = ' ';  // Ensure line will not have correct data if fgets fails
    do {
      if ( ! fgets (line, sizeof line, fp[n])) {
	printf ("Never found 'name' in %s\n", argv[optind+n]);
	return -1;
      }
      ++item[n].nlines_hdr;
    } while (strncmp (line, "name ", 5) != 0);
  }

  difflist = malloc (listmax * sizeof (struct List));
  rdifflist = malloc (listmax * sizeof (struct List));
  listsize_diff = 0;
  listsize_rdiff = 0;

  // Read through the files, comparing "wallmax" entries
  while (fgets (line, sizeof line, fp[0])) {
    ++item[0].nlines_data;
    if ((ret = scanit (line, &item[0])) == 1) {
      printf ("%s: scanit reached EOF on file 0: Printing results:\n", thisfunc);
      break;
    } else if (ret != 6) {
      printf ("%s: Strange return=%d from scanit: Stopping\n", thisfunc, ret);
      return -1;
    }
    if ((ret = get_matching_line (item[0].name, fp[1], &item[1])) > 0) {
      printf ("%s: Failed to match name=%s in file=%s: skipping\n", 
	      thisfunc, item[0].name, argv[optind+1]);
      continue;
    } else if (ret < 0) {
      printf ("%s: fatal error from get_matching_line: stopping\n", thisfunc);
      return -1;
    }
#ifdef DEBUG
    printf ("Found matching line name=%s vals=%f %f\n", 
	    item[0].name, item[0].wallmax, item[1].wallmax);
#endif
    walldiff = fabsf (item[0].wallmax - item[1].wallmax);
    if (walldiff > mindiff) {
      maybe_insert (walldiff, difflist, item, &listsize_diff);
      // Avoid division by zero
      if (item[0].wallmax != item[1].wallmax) {
	reldiff = walldiff / (0.5*(item[0].wallmax + item[1].wallmax));
	maybe_insert (reldiff, rdifflist, item, &listsize_rdiff);
      }
    }
  }

  printlist ("Difflist", "diff ", difflist, listsize_diff);
  printlist ("\nRelative difflist", "rdiff", rdifflist, listsize_rdiff);
  return 0;
}
  
int get_matching_line (const char *name2match, FILE *fp, struct Item *item)
{
  int n;
  int nitems;
  int nlines_data_save;   // save the initial value of nlines_data if need to replace
  char line[MAXSIZ_LINE]; // input line from file
  static const char *thisfunc = "get_matching_line";

  nlines_data_save = item->nlines_data;

  // First try reading ahead
  while (fgets (line, sizeof line, fp)) {
    ++item->nlines_data;
    if (scanit (line, item) == 6) {
      if (strcmp (item->name, name2match) == 0) {
#ifdef DEBUG
	printf ("found it ahead at line=%d wallmax=%f\n",
		item->nlines_hdr+item->nlines_data, item->wallmax);
#endif
	return 0;
      }
    } else if (strncmp (line, "done", 4) == 0) {
      printf ("%s: scanit reached EOF. Next check ...\n", thisfunc);
    } else {
      printf ("%s: Bad return from scanit. line=%s\n", thisfunc, line);
      return -1;
    }
  }
  
  // Not found: Try rewinding and start from the beginning
#ifdef DEBUG
  printf ("%s: %s not found ahead in 2nd file. Rewinding at starting from top...\n",
	  thisfunc, name2match);
#endif
  if (skipjunk (fp, item->nlines_hdr, line) < 0) {
    printf ("%s: skipjunk failure\n", thisfunc);
    return -1;
  }

  // Reality check
  if (strncmp (line, "name ", 5) != 0) {
    printf ("%s: expected 'name' got %5s\n", thisfunc, line);
    return -1;
  }

  // Read through the file looking for matching 'name', at most up until where we started the search
  // Reset nlines_data to 0 so if matching entry is found, it will be reset to the correct value
  item->nlines_data = 0;
  for (n = 0; n < nlines_data_save; ++n) {
    if ( ! fgets (line, sizeof line, fp)) {
      printf ("%s: Unexpected failure from fgets\n", thisfunc);
      return -1;
    }
    ++item->nlines_data;
    if (scanit (line, item) != 6) { // probably reached EOF
      if (skipjunk (fp, item->nlines_hdr, line) < 0) {
	printf ("%s: skipjunk failure\n", thisfunc);
	return -1;
      }
      break;
    }
    if (strcmp (item->name, name2match) == 0) {
#ifdef DEBUG
      printf ("found it behind at line=%d wallmax=%f\n",
	      item->nlines_hdr+item->nlines_data, item->wallmax);
#endif
      return 0;
    }
  }
  printf ("%s: %s not found: skipping\n", thisfunc, name2match);
  return 1;
}

void maybe_insert (float diff, struct List *list, struct Item item[2], int *listsize)
{
  int n, nn;

#ifdef DEBUG
  printf ("\ntesting diff=%f item=%s vals=%f %f\n", diff, item[0].name, item[0].wallmax, item[1].wallmax);
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
    list[n].value[0] = item[0].wallmax;
    list[n].value[1] = item[1].wallmax;
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
    printf ("n=%4d %s=%9.3f values=%9.3f %9.3f ratio (a/b) =%7.3f name=%s\n",
	    n, difftype, list[n].diff, list[n].value[0], list[n].value[1], list[n].value[0]/list[n].value[1], list[n].name);
  }
}

int scanit (const char *line, struct Item *item)
{
  int nitems;

  nitems = sscanf (line, "%s %f %d %f %f %f",
		   item->name, &(item->ncalls), &(item->nranks), &(item->mean_time),
		   &(item->std_dev), &(item->wallmax));
  if (nitems != 6) {
    printf ("scanit: sscanf returned %d items expected 6: maybe timer name has embedded spaces?\n",
	    nitems);
    if (nitems > 0)
      printf ("item 0 got %s\n", item->name);
    if (nitems > 1)
      printf ("item 1 got %f\n", item->ncalls);
  }
  return nitems;
}

int skipjunk (FILE *fp, int nlines, char line[MAXSIZ_LINE])
{
  int n;
  static const char *thisfunc = "skipjunk";

#ifdef DEBUG
  printf ("%s: skipping %d header lines from fp=%d\n", thisfunc, nlines, (int) fp);
#endif
  rewind (fp);
  // Read to the first data item
  for (n = 0; n < nlines; ++n) {
    if ( ! fgets (line, MAXSIZ_LINE, fp)) {
      printf ("%s: failed reading header info\n", thisfunc);
      return -1;
    }
#ifdef DEBUG
    printf ("%s: header line %d starts with %5s\n", thisfunc, n, line);
#endif
  }
  return 0;
}
