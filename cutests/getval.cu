#include <stdio.h>

__host__ int getval_int (const char *str, const int arg)
{
  int ans;
  int ret = arg;

  printf ("Enter %s or -1 to accept default (%d)\n", str, arg);
  scanf ("%d", &ans);
  if (ans != -1)
    ret = ans;
  printf ("arg=%d\n", arg);
  return ret;
}

