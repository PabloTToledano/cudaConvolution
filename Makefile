CXX=nvcc

CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv4`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib -lcudart `pkg-config --libs opencv4`

#Uncomment the line below if you dont have CUDA enabled GPU
#EMU=-deviceemu

ifdef EMU
CUDAFLAGS+=-deviceemu
endif

all:
	$(CXX) $(CFLAGS) -c src/main.cpp -o Debug/main.o
	$(CXX) $(CFLAGS) -c src/cudaFilter.cpp -o Debug/cudaFilter.o
	nvcc $(CUDAFLAGS) -c src/cudaKernels.cu -o Debug/cudaKernels.o
	$(CXX) $(LDFLAGS) Debug/main.o Debug/cudaKernels.o Debug/cudaFilter.o -o Debug/cudaConvolution

clean:
	rm -f Debug/*.o Debug/cudaFilters

