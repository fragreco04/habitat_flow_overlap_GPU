CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -I..
LDLIBS = -lm

SRCS = overlap_cpu_mat.c ../cJSON.c
DEPS = my_utilities.h ../cJSON.h

all: overlap_cpu_mat

overlap_cpu_mat: $(SRCS) $(DEPS)
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDLIBS)

clean:
	rm -f overlap_cpu_mat

.PHONY: all clean