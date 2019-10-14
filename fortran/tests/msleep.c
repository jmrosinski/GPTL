#include <unistd.h>

int msleep_ (int *msec)
{
  return usleep ((*msec) * 1000);
}
