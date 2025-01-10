BOOST_HOME = /Users/dekelsch/Software/boost_1_83_0
BLAS_HOME = /Users/dekelsch/brew-4.1.24/Cellar/openblas/0.3.28
CXXFLAGS += -std=c++17 -I$(BOOST_HOME)/include -I$(BLAS_HOME)/include
LDLIBS += -lm -llapack -lblas

CC = mpic++

