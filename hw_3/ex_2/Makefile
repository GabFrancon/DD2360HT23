# missing includes
MSVC_INCLUDE="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.35.32215/include"
CUDA_INCLUDE="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include"
UCRT_INCLUDE="C:/Program Files (x86)/Windows Kits/10/Include/10.0.22000.0/ucrt"

CXX=cl
CXXFLAGS= /Iinclude /I$(MSVC_INCLUDE) /I$(CUDA_INCLUDE) /I$(UCRT_INCLUDE) /O2 /EHsc /W0

NVCC=nvcc
ARCH=sm_75
NVCCFLAGS= -Iinclude -I$(CUDA_INCLUDE) -arch=$(ARCH) -O3 -g -w

SRCDIR=src
OBJDIR=obj
BINDIR=bin
TARGET=sputniPIC.out

SRCS=$(wildcard $(SRCDIR)/*.cu $(SRCDIR)/*.cpp)
OBJS=$(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.obj,$(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.obj,$(SRCS)))

all: dir $(BINDIR)/$(TARGET)

dir:
	@if not exist $(OBJDIR) mkdir $(OBJDIR)
	@if not exist $(BINDIR) mkdir $(BINDIR)

$(BINDIR)/$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(OBJDIR)/%.obj: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.obj: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) /Fo$@ /c $<

clean:
	del /S /Q $(OBJDIR)
	del /S /Q $(BINDIR)
