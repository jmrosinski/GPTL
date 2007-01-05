#include <unistd.h>  /* getopt */
#include <stdlib.h>  /* atof */
#include "../gptl.h"

int main (int argc, char **argv)
{
  extern char *optarg;
  int c;
  int papiopt;
  MPI_Status status;

  MPI_Init (&argc, &argv);
  ret = MPI_Comm_rank (MPI_COMM_WORLD, &iam);
  ret = MPI_Comm_size (MPI_COMM_WORLD, &size);

  while ((c = getopt (argc, argv, "f:m:o:p:")) != -1) {
    switch (c) {
	case 'f':
	  flopspermem = atof (optarg);
	  printf ("flops per mem ref will be %f\n", flopspermem);
	  break;
	case 'm':
	  flopspersend = atof (optarg);
	  printf ("flops per floating pt vals sent will be %d\n", flopspersend);
	  break;
	case 'o':
	  nompiter = atoi (optarg);
	  printf ("Set nompiter=%d\n", nompiter);
	  break;
	case 'p':
	  if ((papiopt = GPTL_PAPIname2id (optarg)) >= 0) {
	    printf ("Failure from GPTL_PAPIname2id\n");
	    exit (1);
	  }
	  if (GPTLsetoption (papiopt, 1) < 0) {
	    printf ("Failure from GPTLsetoption (%s,1)\n", optarg);
	    exit (1);
	  }
	  break;
	default:
	  printf ("unknown option %c\n", c);
	  exit (2);
    }
  }

  sendbuf = (float *) malloc (sendsize * sizeof (float));

  for (i = 0; i < sendcount; i++)
    sendbuf[i] = iam + i;

  heis = (iam + 1) % size;  /* neighbor */
  ret = MPI_Sendrecv (sendbuf, sendcount, MPI_FLOAT, heis, sendtag,
		      recvbuf, recvcount, MPI_FLOAT, heis, recvtag,
		      MPI_COMM_WORLD, &status);

  for (i = 0; i < recvcount; i++) {
    expect = heis + i;
    rdiff = (expect - recvbuf[i]) / (0.5*(expect + recvbuf[i]));

    if (rdiff > maxrdiff)
      maxrdiff = rdiff;
  }
}
