ifeq "$(OS)" ""
OS = $(shell uname -s)
endif

ifeq "$(NODE)" ""
NODE = $(shell uname -n)
endif

EIGENINC = /Volumes/Files/GitHub/eigen
LAPACK = -lblas -llapack -lm -larpack

CC = clang++ -O3 -m64 -std=c++11 -stdlib=libc++ -I./ -I$(EIGENINC)

.PHONY: all clean

all: build/arpack.app build/zeigh.app build/deigh.app

build/lapack.o: lapack/lapack.cpp
	$(CC) -c -o $@ $<

build/arpack.app: arpack.cpp build/lapack.o
	$(CC) -o $@ $< build/lapack.o $(LAPACK)

build/zeigh.app: zeigh.cpp build/lapack.o
	$(CC) -o $@ $< build/lapack.o $(LAPACK)

build/deigh.app: deigh.cpp build/lapack.o
	$(CC) -o $@ $< build/lapack.o $(LAPACK)
