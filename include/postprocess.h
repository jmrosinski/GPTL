#include "private.h"

namespace postprocess {
  extern Settings overheadstats;
  extern GPTLMethod method;
  extern bool percent;
  extern bool dopr_preamble;
  extern bool dopr_threadsort;
  extern bool dopr_multparent;
  extern bool dopr_collision;

  extern "C" {
    char *methodstr (GPTLMethod);
  }
}
