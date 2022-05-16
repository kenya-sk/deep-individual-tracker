CC := g++
CFLAG := -O3
CFLAG += -std=c++11
PKGCONFIG = `pkg-config --cflags opencv` `pkg-config --libs opencv`
TARGET := main
SRCS := main.cpp
SRCS += tracker.cpp
SRCS += timer.cpp
SRCS += tracking_config.cpp
SRCS += utils.cpp
OBJS := $(SRCS:.cpp=.o)


$(TARGET): $(OBJS)
	$(CC) $(CFLAG) -o $@ $^ $(PKGCONFIG)

.PHONY: all
all: clean $(TARGET)

.PHONY: clean
clean:
	-rm -f $(OBJS) $(TARGET) *.d