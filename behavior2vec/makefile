CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -g -march=native -Wall -funroll-loops -Wno-unused-result -O3

all: main
#word2phrase distance word-analogy compute-accuracy

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
main: main.c
	$(CC) main.c -o main $(CFLAGS)
deepbehavior: deepbehavior.c
	$(CC) deepbehavior.c -o deepbehavior $(CFLAGS)
#word2phrase : word2phrase.c
#word2phrase : word2phrase.c
#	$(CC) word2phrase.c -o word2phrase $(CFLAGS)
#distance : distance.c
#	$(CC) distance.c -o distance $(CFLAGS)
#word-analogy : word-analogy.c
#	$(CC) word-analogy.c -o word-analogy $(CFLAGS)
#compute-accuracy : compute-accuracy.c
#	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
#	chmod +x *.sh

clean:
	rm -rf main
#word2phrase distance word-analogy compute-accuracy
