CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2
INCLUDES  = -I../src

all: test-philox test-savgol

test-philox: test-philox.cpp ../src/philox.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< -lm

test-savgol: test-savgol.cpp ../src/savgol.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< -lm

clean:
	rm -f test-philox philox-noise.f32 test-savgol

.PHONY: all clean
