CC = g++
#CC = clang++

CFLAGS = -std=c++17 -Wall -g -O3 -fopenmp -march=native #-Wextra -MD -MP

# include the paths to Eigen and hdf5 
#INCLUDES = -I./libs/eigen-3.3.9/ -I/u/22/pyykkov2/unix/anaconda3/include
INCLUDES = -I../libs/eigen-3.3.9/ -I/home/ville/anaconda3/include

LIBS = -lhdf5_cpp -lhdf5

# add the linker paths for hdf5
#LFLAGS = -Wl,-rpath=/u/22/pyykkov2/unix/anaconda3/lib -L/u/22/pyykkov2/unix/anaconda3/lib
LFLAGS = -Wl,-rpath=/home/ville/anaconda3/lib -L/home/ville/anaconda3/lib

SRCS = $(wildcard *.cpp)

SRCS = SSJunction.cpp TwoTerminalSetup.cpp ScatteringSystem.cpp Lead.cpp fd_dist.cpp ScfSolver.cpp ScfMethod.cpp da2glob.cpp Global.cpp Interval.cpp Observable.cpp pade_frequencies.cpp config_parser.cpp quadrature.cpp file_io.cpp connectivity.cpp

SRC1 = prepare_datapoints.cpp
SRC2 = process_datapoint.cpp

OBJS = $(SRCS:.cpp=.o)
OBJ1 = $(SRC1:.cpp=.o)
OBJ2 = $(SRC2:.cpp=.o)


ALL_OBJS = $(OBJS) $(OBJ1) $(OBJ2) 
depends = $(all_obj:%.o=%.d)

MAIN1 = prepare_datapoints
MAIN2 = process_datapoint

.PHONY: depend clean


all:	$(MAIN1) $(MAIN2) 
		@echo Everything went better than expected

prepare: $(MAIN1)
process: $(MAIN2)


$(MAIN1): $(OBJ1) $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) $(LFLAGS) -o $(MAIN1) $(OBJ1) $(OBJS) $(LIBS)


$(MAIN2): $(OBJ2) $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) $(LFLAGS) -o $(MAIN2) $(OBJ2) $(OBJS) $(LIBS)



%.o: %.cpp Makefile
	$(CC) $(CFLAGS) $(INCLUDES) $(LFLAGS)   -c $< -o $@ $(LIBS)

clean:
	$(RM) *.o *~ *.d $(MAIN1) $(MAIN2) 

-include $(DEPENDS)

#depend: $(SRCS)
# 		makedepend $(INCLUDES) $^
