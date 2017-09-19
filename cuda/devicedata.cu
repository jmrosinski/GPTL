#include "./private.h"

__device__ Timer **timers = 0;             /* linked list of timers */
__device__ Timer **last = 0;               /* last element in list */
__device__ int *max_name_len;              /* max length of timer name */
__device__ int maxthreads = -1;            /* max threads */
__device__ int maxwarps = -1;              /* max warps */
__device__ int nwarps_found = 0;           /* number of warps found : init to 0 */
__device__ int nwarps_timed = 0;           /* number of warps analyzed : init to 0 */
__device__ bool disabled = false;          /* Timers disabled? */
__device__ bool initialized;       /* GPTLinitialize has been called */
__device__ bool verbose = false;           /* output verbosity */

/* Options, print strings, and default enable flags */
__device__ Hashentry **hashtable;          /* table of entries */
__device__ int tablesize;                  // size of hash table
__device__ int tablesizem1;                // one less
