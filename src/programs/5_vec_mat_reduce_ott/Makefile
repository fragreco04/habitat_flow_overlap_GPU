CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
LDLIBS = -lOpenCL -lm

TARGET = overlap_vec_mat_reduce_batch
SRCS = $(TARGET).c ../../modules/cJSON.c
DEPS = my_utilities.h ocl_boiler.h ../../modules/cJSON.h ../../modules/info.h

all: $(TARGET)

$(TARGET): $(SRCS) $(DEPS)
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDLIBS)

# Target per pulire
clean:
	rm -f $(TARGET)

.PHONY: all clean