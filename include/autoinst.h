#ifndef AUTOINST_H
#define AUTOINST_H

#include "private.h"

namespace gptl_autoinst {
  using namespace gptl_private;
  extern "C" {
    inline Timer *getentry_instr (const Hashentry *, void *, unsigned int *);
  }
};
#endif
