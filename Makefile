# SPDX-License-Identifier: AGPL-3.0-or-later
# RPI — Resonant Permutation Inference
# (c) 2026 Elyan Labs

CC ?= gcc
# -pthread in BOTH compile and link: compiling with it sets _REENTRANT /
# thread-safe libc paths on platforms that need it, not just the link flag.
CFLAGS = -O3 -Wall -Wextra -I include -pthread
LDFLAGS = -pthread

# Auto-detect architecture
UNAME_M := $(shell uname -m)

ifeq ($(findstring ppc,$(UNAME_M)),ppc)
  ifeq ($(findstring ppc64,$(UNAME_M)),ppc64)
    CFLAGS += -mcpu=power8 -maltivec -mvsx -DRPI_POWER8
    $(info Building for POWER8 (VSX + AltiVec))
  else
    CFLAGS += -mcpu=7450 -maltivec -DRPI_G4
    $(info Building for PowerPC G4 (AltiVec))
  endif
else ifeq ($(findstring x86_64,$(UNAME_M)),x86_64)
  CFLAGS += -march=native -DRPI_X86
  $(info Building for x86_64)
else ifneq ($(filter arm64 aarch64,$(UNAME_M)),)
  # macOS reports "arm64", Linux reports "aarch64". NEON tbl is baseline
  # on AArch64, so no -mfpu is needed; -mcpu just improves scheduling.
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    # Overridable: `make ARM_CPU=apple-m2` (or native). apple-m1 is a safe
    # baseline accepted by older clang and correct on every M-series chip.
    ARM_CPU ?= apple-m1
    CFLAGS += -mcpu=$(ARM_CPU) -DRPI_ARM64
    $(info Building for Apple Silicon (NEON tbl, -mcpu=$(ARM_CPU)))
  else
    CFLAGS += -march=native -DRPI_ARM64
    $(info Building for AArch64 (NEON tbl))
  endif
endif

SRCS = src/common/model.c src/common/decode.c src/main.c

# Add platform-specific sources
ifeq ($(findstring ppc64,$(UNAME_M)),ppc64)
  SRCS += src/power8/perm_vsx.c
else ifeq ($(findstring ppc,$(UNAME_M)),ppc)
  SRCS += src/g4/perm_altivec.c
else ifneq ($(filter arm64 aarch64,$(UNAME_M)),)
  SRCS += src/arm64/perm_neon.c
endif
OBJS = $(SRCS:.c=.o)

.PHONY: all clean test

all: rpi-cli

rpi-cli: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: rpi-cli
	python3 tools/gen_test_model.py
	./rpi-cli -m test_model.rpi -p "hello" -n 20 -v

clean:
	rm -f $(OBJS) rpi-cli test_model.rpi
