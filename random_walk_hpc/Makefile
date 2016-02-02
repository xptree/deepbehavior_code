CC=g++
CFLAGS= -O0 -g -Wall -std=c++11 -fopenmp  -I/home/jiezhong/local/include
LDFLAGS= -O0 -Wall -std=c++11 -fopenmp -llog4cpp -L/home/jiezhong/local/lib
EXEC=main
OBJ= Entity.o Action.o Behavior.o History.o Util.o Graph.o main.o Parameter.o

all: $(EXEC)

main : $(OBJ)
		$(CC) -o $@ $^ $(LDFLAGS)

##########################################
# Generic rules
# ##########################################
#
%.o: %.cpp %.h
	$(CC) -o $@ -c $< $(CFLAGS)

%.o: %.cpp
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -f *.o *~ $(EXEC)
