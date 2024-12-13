OBJ = $(SRC)/*.cpp

SRC = src
BIN = bin
MAIN_OBJ = 1D_Euler_Sod_try2.cpp
EXEC = 1d_burgers
TESTBIN = tests/bin
TEST1 = tests/test1
TEST2 = tests/test2
TEST3 = tests/test3
CXXFLAGS += -std=c++17 -g -DMPI_NO_CPPBIND
include module.mk

all:		install
	
install:	makebin
		$(CC) $(CXXFLAGS) $(OBJ) $(MAIN_OBJ) -o $(BIN)/$(EXEC) $(LDFLAGS) $(LDLIBS)
		cp -r $(BIN)/$(EXEC) .

makebin:
		mkdir -p $(BIN)
test:
		make -C $(TEST1)
		make -C $(TEST2)
		make -C $(TEST3)
clean:
		rm -rf $(EXEC) *.dat $(SRC)/*.o $(SRC)/*.mod $(BIN) $(TESTBIN) grid.h5
