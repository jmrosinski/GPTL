#include <openacc.h>
#include <stdio.h>

// TBD: Examine "tile" clause
// TBD: Examine "cache" clause
// TBD: Examine "bind" clause
// TBD: Examine "nohost" clause
int main ()
{
  int nsm = 5;
  int nwarps = 4;
  int warpsize = 32;
  int ndev;
  int devnum;
  acc_device_t devtyp;
  int sum = 0;
  int ret;
  const char *str;

  printf ("_OPENACC=%d\n", _OPENACC);
  
  devtyp = acc_get_device_type ();
  printf ("devtyp = %d\n", (int) devtyp);

  ndev = acc_get_num_devices (devtyp);
  printf ("ndev=%d\n", ndev);

  devnum = acc_get_device_num (devtyp);
  printf ("devnum = %d\n", (int) devnum);

  acc_init (devtyp);

  str = acc_get_property_string (devnum, devtyp, acc_property_vendor);
  printf ("vendor=%s\n", str);

  str = acc_get_property_string (devnum, devtyp, acc_property_name);
  printf ("device name=%s\n", str);

  str = acc_get_property_string (devnum, devtyp, acc_property_driver);
  printf ("driver=%s\n", str);

#pragma acc parallel
  {
    if (acc_on_device (acc_device_not_host))
      printf ("Test acc parallel region is running on NOT host\n");
    else
      printf ("Test acc parallel region is running on host\n");
  }
  
#pragma acc parallel copy(sum) copyin(nsm,nwarps,warpsize)
  {
#pragma acc loop gang reduction(+:sum)
    for (int g = 0; g < nsm; ++g) {
#pragma acc loop worker reduction(+:sum)
      for (int w = 0; w < nwarps; ++w) {
#pragma acc loop vector reduction(+:sum)
	for (int t = 0; t < warpsize; ++t) {
	  sum += 1;
	}
      }
    }
  }
  
  if ((ret = acc_async_test_all ())) {
    printf ("All async operations have completed\n");
  } else {
    printf ("Some async operations still outstanding\n");
    acc_wait_all ();
    printf ("Done now\n");
  }
  printf ("sum=%d\n", sum);
  acc_shutdown (devtyp);
}
