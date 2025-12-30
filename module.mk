BOOST_HOME = /Users/dekelsch/Software/boost_1_83_0
BLAS_HOME = /opt/homebrew/Cellar/openblas/0.3.30
CXXFLAGS += -std=c++17 -I$(BOOST_HOME)/include -I$(BLAS_HOME)/include
LDLIBS += -lm -llapack -lblas

CC = mpic++

