#include "private.h"

namespace postprocess {
  Settings overheadstats;
  GPTLMethod method;
  bool percent;
  bool dopr_preamble;
  bool dopr_threadsort;
  bool dopr_multparent;
  bool dopr_collision;

  extern "C" {
    char *methodstr (GPTLMethod);
  }
}
