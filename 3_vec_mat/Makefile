CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -I..
LDLIBS = -lOpenCL -lm

SRCS = overlap_vec_mat.c ../cJSON.c
DEPS = my_utilities.h ocl_boiler.h ../cJSON.h ../info.h

all: overlap_vec_mat

overlap_vec_mat: $(SRCS) $(DEPS)
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDLIBS)

clean:
	rm -f overlap_vec_mat

.PHONY: all clean