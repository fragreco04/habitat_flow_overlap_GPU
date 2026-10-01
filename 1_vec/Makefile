CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
LDLIBS = -lOpenCL -lm

TARGET = overlap_vec
SRCS = $(TARGET).c
DEPS = my_utilities.h ocl_boiler.h ../info.h

all: $(TARGET)

$(TARGET): $(SRCS) $(DEPS)
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDLIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean