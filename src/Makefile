# When gol-par.c has been created, uncomment the next line:
all: gol-seq gol-par gol-parb1 gol-parb2

gol-seq: gol-seq.c
	gcc -Wall -O3 -o gol-seq gol-seq.c -lm

gol-par: gol-par.c
	mpicc -Wall -O3 -o gol-par gol-par.c -lm 

gol-parb1: gol-parb1.c
	mpicc -Wall -O3 -o gol-parb1 gol-parb1.c -lm 

gol-parb2: gol-parb2.c
	mpicc -Wall -O3 -o gol-parb2 gol-parb2.c -lm

clean:
	rm -f *.o gol-seq gol-par gol-parb1 gol-parb2 *~ *core
