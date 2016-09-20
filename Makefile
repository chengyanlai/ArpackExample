ifeq "$(OS)" ""
OS = $(shell uname -s)
endif

ifeq "$(NODE)" ""
NODE = $(shell uname -n)
endif

ifeq ("$(OS)", "Darwin")
	EIGENINC = /Volumes/Files/GitHub/eigen
	MKLROOT =
	LAPACK =  -lblas -llapack -lm -larpack
	CC = clang++ -O3 -m64 -std=c++11 -stdlib=libc++ -I./ -I$(EIGENINC)
else ifeq ("$(NODE)", "kagome.rcc.ucmerced.edu")
	EIGENINC = /usr/local/include
	MKLROOT = /condensate1/intel/composer_xe_2015.2.164/mkl
	LAPACK = $(MKLROOT)/lib/intel64/libmkl_blas95_lp64.a \
	$(MKLROOT)/lib/intel64/libmkl_lapack95_lp64.a -Wl,--start-group \
	$(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
	$(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a \
	-Wl,--end-group -lpthread -lm -larpack
	CC = icpc -O3 -Wall -std=c++11 -I./ -I$(EIGENINC) -DMKL
else ifeq ("$(NODE)", "edgestate.rcc.ucmerced.edu")
	EIGENINC = /usr/local/include
	MKLROOT = /opt/intel/composer_xe_2015.2.164/mkl
	LAPACK = $(MKLROOT)/lib/intel64/libmkl_blas95_lp64.a \
	$(MKLROOT)/lib/intel64/libmkl_lapack95_lp64.a -Wl,--start-group \
	${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
	${MKLROOT}/lib/intel64/libmkl_core.a \
	${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm -larpack
	CC = icpc -qopenmp -O3 -Wall -std=c++11 -I./ -I$(EIGENINC) -DMKL
endif

BUILD_DIR = ./build

.PHONY: checkdirs all clean

all: checkdirs build/arpack.app build/zeigh.app build/deigh.app

build/lapack.o: lapack/lapack.cpp
	$(CC) -c -o $@ $<

build/arpack.app: arpack.cpp build/lapack.o
	$(CC) -o $@ $< build/lapack.o $(LAPACK)

build/zeigh.app: zeigh.cpp build/lapack.o
	$(CC) -o $@ $< build/lapack.o $(LAPACK)

build/deigh.app: deigh.cpp build/lapack.o
	$(CC) -o $@ $< build/lapack.o $(LAPACK)

checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(BUILD_DIR) build/*.o
