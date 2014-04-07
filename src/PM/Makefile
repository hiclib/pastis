# This makefile was adapted from the example from IPOPT.


# CHANGEME: 
# The following is the directory where IPOPT is installed
IPOPTPATH = /home/nelle/.local
# At UW, ARCH is a variable set in ~/.bashrc

# IPOPT directory with header files
IPOPTINCDIR = ${IPOPTPATH}/include/coin

# IPOPT directory with libipopt.a
IPOPTLIBDIR = ${IPOPTPATH}/lib


# The name of the executable
EXE = PM_all

# Here is the name of all object files corresponding to the source
#           code that you wrote in order to define the problem statement
OBJS = coords.o \
       utils.o \
       genome_ipopt_main.o \
       genome_ipopt_nlp.o \
       genome_model.o \
       metric.o \
       frequencies.o

# Additional libraries
ADDLIBS =

# Additional flags for compilation (e.g., include flags)
ADDINCFLAGS =

##########################################################################
#  Usually, you don't have to change anything below.  Note that if you   #
#  change certain compiler options, you might have to recompile Ipopt.   #
##########################################################################

# C++ Compiler command
CXX = g++
# On n001 it is g++4
# on guanine is g++

# C++ Compiler options

# include -NDEBUG if you want to disable the assert functions
CXXFLAGS = -fomit-frame-pointer -pipe -Wimplicit -Wparentheses -Wreturn-type -Wcast-qual -Wall -Wpointer-arith -Wwrite-strings -Wconversion -Wno-unknown-pragmas -g

# additional C++ Compiler options for linking
CXXLINKFLAGS =  -Wl,--rpath -Wl,${IPOPTLIBDIR} -Wl,--no-as-needed


# Libraries necessary to link with IPOPT
LIBS = -L$(IPOPTLIBDIR) -lipopt -lpthread -llapack -lblas -lm  -ldl -lgfortranbegin -lgfortran -lm -lgcc_s -lcoinhsl


# Necessary Include dirs (we use the CYGPATH_W variables to allow
# compilation with Windows compilers)
INCL =  -I`$(CYGPATH_W) $(IPOPTINCDIR)` $(ADDINCFLAGS)

# The following is necessary under cygwin, if native compilers are used
CYGPATH_W = echo

all: $(EXE)

.SUFFIXES: .cpp .c .o .obj

$(EXE): $(OBJS)
	bla=;\
	for file in $(OBJS); do bla="$$bla `$(CYGPATH_W) $$file`"; done; \
	$(CXX) $(CXXLINKFLAGS) $(CXXFLAGS) -o $@ $$bla $(ADDLIBS) $(LIBS)

clean:
	rm -rf $(EXE) $(OBJS) ipopt.out

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCL) -c -o $@ $<


.cpp.obj:
	$(CXX) $(CXXFLAGS) $(INCL) -c -o $@ `$(CYGPATH_W) '$<'`

trailing-spaces:
	find . -name "*.cpp" | xargs perl -pi -e 's/[ \t]*$$//'
	find . -name "*.hpp" | xargs perl -pi -e 's/[ \t]*$$//'
	find . -name "*.h" | xargs perl -pi -e 's/[ \t]*$$//'

tarball:
	git archive --format=zip master . > pm.zip

