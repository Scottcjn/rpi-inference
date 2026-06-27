# SPDX-License-Identifier: AGPL-3.0-or-later
# RPI — Resonant Permutation Inference
# (c) 2026 Elyan Labs

CC ?= gcc
CFLAGS = -O3 -Wall -Wextra -I include
LDFLAGS =

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
else ifeq ($(findstring aarch64,$(UNAME_M)),aarch64)
  CFLAGS += -march=native -DRPI_ARM64
  $(info Building for AArch64)
endif

SRCS = src/common/model.c src/common/decode.c src/main.c

# Add platform-specific sources
ifeq ($(findstring ppc64,$(UNAME_M)),ppc64)
  SRCS += src/power8/perm_vsx.c
else ifeq ($(findstring ppc,$(UNAME_M)),ppc)
  SRCS += src/g4/perm_altivec.c
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
