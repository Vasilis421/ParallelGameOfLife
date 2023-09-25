/***********************

Conway's Game of Life

Based on https://web.cs.dal.ca/~arc/teaching/CS4125/2014winter/Assignment2/Assignment2.html

************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "mpi.h"

typedef struct {
   int height, width;
   int **cells;
} world;

static world worlds[2];
static world *cur_world, *next_world;

static int print_cells = 0;
static int print_world = 0;

// use fixed world or random world?
#ifdef FIXED_WORLD
static int random_world = 0;
#else
static int random_world = 1;
#endif

static char *start_world[] = {
    /* Gosper glider gun */
    /* example from https://bitstorm.org/gameoflife/ */
    "..........................................",
    "..........................................",
    "..........................................",
    "..........................................",
    "..........................................",
    "..........................................",
    "........................OO.........OO.....",
    ".......................O.O.........OO.....",
    ".OO.......OO...........OO.................",
    ".OO......O.O..............................",
    ".........OO......OO.......................",
    ".................O.O......................",
    ".................O........................",
    "....................................OO....",
    "....................................O.O...",
    "....................................O.....",
    "..........................................",
    "..........................................",
    ".........................OOO..............",
    ".........................O................",
    "..........................O...............",
    "..........................................",
};

static void
world_init_fixed(world *world)
{
    int **cells = world->cells;
    int i, j;

    /* use predefined start_world */

    for (i = 1; i <= world->height; i++) {
        for (j = 1; j <= world->width; j++) {
            if ((i <= sizeof(start_world) / sizeof(char *)) &&
                (j <= strlen(start_world[i - 1]))) {
                cells[i][j] = (start_world[i - 1][j - 1] != '.');
            } else {
                cells[i][j] = 0;
            }
        }
    }
}

static void
world_init_random(world *world)
{
    int **cells = world->cells;
    int i, j;

    // Note that rand() implementation is platform dependent.
    // At least make it reprodible on this platform by means of srand()
    srand(1);

    for (i = 1; i <= world->height; i++) {
        for (j = 1; j <= world->width; j++) {
            float x = rand() / ((float)RAND_MAX + 1);
            if (x < 0.5) {
                cells[i][j] = 0;
            } else {
                cells[i][j] = 1;
            }
        }
    }
}

static void
world_print(world *world)
{
    int **cells = world->cells;
    int i, j;

    for (i = 1; i <= world->height; i++) {
        for (j = 1; j <= world->width; j++) {
            if (cells[i][j]) {
                printf("O");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
}

static int
world_count(world *world)
{
    int **cells = world->cells;
    int isum;
    int i, j;

    isum = 0;
    for (i = 1; i <= world->height; i++) {
        for (j = 1; j <= world->width; j++) {
            isum = isum + cells[i][j];
        }
    }

    return isum;
}

void
decide_neighbours(int* top, int* bottom, int rank, int p)
{
    // process 0 sends top row to last process
    // and bottom row to next process
    if (rank == 0) {
	    *top = p - 1;
        if (p == 1) { // in case of one process neighbours are myself 
            *bottom = p - 1;
        } else {
            *bottom = rank + 1;
        }
    // last process sends top row to its previous process
    // and bottom row to process 0
    } else if (rank == p - 1) {
        *top = rank - 1;
        *bottom = 0;
    // rest processes send top row to previous process
    // and bottom row to next process
    } else {
        *top = rank - 1;
        *bottom = rank + 1;
    }
}

/* Take world wrap-around into account: */
static void
world_border_wrap(world *world, int rank, int p)
{
    int **cells = world->cells;
    int j;
    // create the appropriate datatype to send a single column
    MPI_Datatype vector;
    MPI_Type_vector(world->height + 2, 1, world->width + 2, MPI_INT, &vector);
    MPI_Type_commit(&vector);

    /* top-bottom boundary conditions */
    for (j = 1; j <= world->width; j++) {
        cells[0][j] = cells[world->height][j];
        cells[world->height + 1][j] = cells[1][j];
    }

    /* left-right boundary conditions */
    // get the neighbouring processes
    int left_neighbour, right_neighbour;
    decide_neighbours(&left_neighbour, &right_neighbour, rank, p); 
    
    // communicate columns with neighbours
    if (rank % 2 == 0)
    {   // send first real (not ghost) column to left neighbour
        MPI_Send(&world->cells[0][1], 1, vector, 
                 left_neighbour, 2, MPI_COMM_WORLD);
	    // receive right neighbour's first real column and set it as right ghost
        MPI_Recv(&world->cells[0][world->width + 1], 1, vector, 
                 right_neighbour, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    // send last real column to right neighbour
        MPI_Send(&world->cells[0][world->width], 1, vector,
                 right_neighbour, 2, MPI_COMM_WORLD);
	    // receive left neighbour's last real column and set it as left ghost
        MPI_Recv(&world->cells[0][0], 1, vector,
                 left_neighbour, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else { // odd ranks receive first
        MPI_Recv(&world->cells[0][world->width + 1], 1, vector, 
                 right_neighbour, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&world->cells[0][1], 1, vector, 
                 left_neighbour, 2, MPI_COMM_WORLD);
        MPI_Recv(&world->cells[0][0], 1, vector,
                 left_neighbour, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&world->cells[0][world->width], 1, vector,
                 right_neighbour, 2, MPI_COMM_WORLD);
    }
    MPI_Type_free(&vector);
}

static int
world_cell_newstate(world *world, int row, int col)
{
    int **cells = world->cells;
    int row_m, row_p, col_m, col_p, nsum;
    int newval;

    // sum surrounding cells
    row_m = row - 1;
    row_p = row + 1;
    col_m = col - 1;
    col_p = col + 1;

    nsum = cells[row_p][col_m] + cells[row_p][col] + cells[row_p][col_p]
         + cells[row  ][col_m]                     + cells[row  ][col_p]
         + cells[row_m][col_m] + cells[row_m][col] + cells[row_m][col_p];

    switch (nsum) {
    case 3:
        // a new cell is born
        newval = 1;
        break;
    case 2:
        // nothing happens
        newval = cells[row][col];
        break;
    default:
        // the cell, if any, dies
        newval = 0;
    }

    return newval;
}


// update board for next timestep
// height/width params are the base height/width
// excluding the surrounding 1-cell wraparound border
static void
world_timestep(world *old, world *new)
{
    int i, j;

    // update board
    for (i = 1; i <= new->height; i++) {
        for (j = 1; j <= new->width; j++) {
            new->cells[i][j] = world_cell_newstate(old, i, j);
        }
    }
}

static int **
alloc_2d_int_array(int nrows, int ncolumns)
{
    int **array;
    int i;

    /* version that keeps the 2d data contiguous, can help caching and slicing across dimensions */
    array = malloc(nrows * sizeof(int *));
    if (array == NULL) {
       fprintf(stderr, "out of memory\n");
       exit(1);
    }

    array[0] = malloc(nrows * ncolumns * sizeof(int));
    if (array[0] == NULL) {
       fprintf(stderr, "out of memory\n");
       exit(1);
    }

    for (i = 1; i < nrows; i++) {
	array[i] = array[0] + i * ncolumns;
    }

    return array;
}

static double
time_secs(void)
{
    struct timeval tv;

    if (gettimeofday(&tv, 0) != 0) {
        fprintf(stderr, "could not do timing\n");
        exit(1);
    }

    return tv.tv_sec + (tv.tv_usec / 1000000.0);
}


void
grid_decomposition(int width, int rank, int p, 
                   int* subworld_start, int* subworld_width) 
{
	*subworld_start = width / p * rank;
	*subworld_width = width / p;
	int remainder = width % p;
	// give one more column to some processes
	if (rank < remainder) {
	    *subworld_width += 1;
	}
	// adjust starting columns
	if (rank > remainder) {
	    *subworld_start += remainder;
	} else {
	    *subworld_start += rank;
	}
}

void
split_board(world *whole_world, world *subworld, 
            int p, int rank, int start_row, int *widths, int *starts)
{
    // create the datatype to send columns (process 0)
    MPI_Datatype sendvector, sendcolumn;
    MPI_Type_vector(whole_world->height + 2, 1, whole_world->width + 2, MPI_INT, &sendvector);
    // modify the datatype for the columns to be contiguous
    MPI_Type_create_resized(sendvector, 0, sizeof(int), &sendcolumn);
    MPI_Type_commit(&sendcolumn);
    // create the datatype to receive columns
    MPI_Datatype recvvector, recvcolumn;
    MPI_Type_vector(subworld->height + 2, 1, subworld->width + 2, MPI_INT, &recvvector);
    // modify again
    MPI_Type_create_resized(recvvector, 0, sizeof(int), &recvcolumn);
    MPI_Type_commit(&recvcolumn);

    // recv buffer
    int receive[2 * (p - 1)];
    // process 0
    if (rank == 0)
    {
        widths[0] = subworld->width;
        starts[0] = start_row + 1;
	
        for (int n = 1; n < p; n++)
        {   
            // receive from each process their starting column 
            // and the number of columns they will work on
            MPI_Recv(&receive[2 * (n - 1)], 2, MPI_INT, 
                     n, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }    
        // put the received info in their corresponding array (widths and starts)
        for (int i = 0; i < 2 * (p - 1); i++)
        {
            if (i % 2 == 0) {
                starts[i / 2 + 1] = 1 + receive[i];
            } else {
                widths[(int)floor(i / 2) + 1] = receive[i];
            }
        }

        // send to each process its part of the whole_world
        MPI_Scatterv(&whole_world->cells[0][0], widths,
                     starts, sendcolumn, &cur_world->cells[0][1],
                     widths[0], recvcolumn, 0, MPI_COMM_WORLD);
    // other processes
    } else {
        // send my subworld_start and subworld_width to process 0
        int send[2] = {start_row, subworld->width};
        MPI_Send(&send, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
        // reveive my part of the whole_world, 
        // and save it to my current world
        MPI_Scatterv(NULL, NULL, NULL, sendcolumn, &cur_world->cells[0][1], 
                     subworld->width, recvcolumn, 0, MPI_COMM_WORLD);
    }
    MPI_Type_free(&sendcolumn);
    MPI_Type_free(&recvcolumn);
}

void
gather_updated_parts(world *whole_world, world *subworld,
                           int rank, int p, int *widths, int *starts)
{
    MPI_Datatype sendvector, sendcolumn, recvvector, recvcolumn;
    MPI_Type_vector(subworld->height + 2, 1, subworld->width + 2, 
                                           MPI_INT, &sendvector);
    MPI_Type_create_resized(sendvector, 0, sizeof(int), &sendcolumn);
    MPI_Type_commit(&sendcolumn);
    
    MPI_Type_vector(whole_world->height + 2, 1, whole_world->width + 2, 
                                                 MPI_INT, &recvvector);
    MPI_Type_create_resized(recvvector, 0, sizeof(int), &recvcolumn);
    MPI_Type_commit(&recvcolumn);
    
    // process 0 reassembles updated world for printing purposes
    if (rank == 0)
    {	
        MPI_Gatherv(&subworld->cells[0][1], widths[0], sendcolumn, 
                        &whole_world->cells[0][0], widths, starts, 
                                   recvcolumn, 0, MPI_COMM_WORLD);
    // other processes send their updated parts to process 0
    } else {
        MPI_Gatherv(&cur_world->cells[0][1],
                    subworld->width, sendcolumn, 
                    NULL, NULL, NULL, recvcolumn, 0, MPI_COMM_WORLD);
    }
    MPI_Type_free(&sendcolumn);
    MPI_Type_free(&recvcolumn);
}

int
main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    int number_of_processors;
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processors);

    int processor_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);

    int n, nsteps;
    double start_time, end_time, elapsed_time;
    int bwidth, bheight;

    /* get parameters */
    if (argc != 6) {
        if (processor_rank == 0) {
            fprintf(stderr, "Usage: %s width height steps print_world print_cells\n", argv[0]);
        }
        exit(1);
    }
    
    bwidth = atoi(argv[1]);
    bheight = atoi(argv[2]);
    nsteps = atoi(argv[3]);
    print_world = atoi(argv[4]);
    print_cells = atoi(argv[5]);

    if (bwidth < number_of_processors) {
        if (processor_rank == 0) {
            fprintf(stderr, "The world width must be equal to or greater than the number of processes.\n");
        }
        exit(1);
    }

    // decide the columns each process will work on
    int subworld_start, subworld_width;
	grid_decomposition(bwidth, processor_rank, number_of_processors,
					   &subworld_start, &subworld_width);
    
    /* initialize worlds, when allocating arrays, add 2 for ghost cells in both directorions */
    // only their part of the grid is allocated in each process to save memory
    worlds[0].height = bheight;
    worlds[0].width = subworld_width;
    worlds[0].cells = alloc_2d_int_array(bheight + 2, subworld_width + 2);

    worlds[1].height = bheight;
    worlds[1].width = subworld_width;
    worlds[1].cells = alloc_2d_int_array(bheight + 2, subworld_width + 2);

    // these are now the subworlds
    cur_world = &worlds[0];
    next_world = &worlds[1];

    // initialize whole world for printing purposes (only in process 0)
    static world whole_world;
    if (processor_rank == 0) 
    {
        whole_world.height = bheight;
        whole_world.width = bwidth;
        whole_world.cells = alloc_2d_int_array(bheight + 2, bwidth + 2);
    }
    
    // initialize random or ficex board in process 0 
    if (processor_rank == 0)
    {
        if (random_world) {
            world_init_random(&whole_world);
        } else {
            world_init_fixed(&whole_world);
        }
    }
    // arrays to save the number of columns and starting column of each process
    // (needed for MPI_Scatterv and MPI_Gatherv)
    int widths[number_of_processors], starts[number_of_processors];
    
    // send to each process its part
    split_board(&whole_world, cur_world, number_of_processors, 
                processor_rank, subworld_start, widths, starts);
    
    // only process 0 prints
    if (print_world > 0 && processor_rank == 0) {
        printf("\ninitial world:\n\n");
        world_print(&whole_world);
    }
    // barrier for measurement purposes
    // so that all processes start at the same time
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = time_secs();

    /* time steps */
    for (n = 0; n < nsteps; n++) {
        world *tmp_world;

        world_border_wrap(cur_world, processor_rank, number_of_processors);
        world_timestep(cur_world, next_world);

        // swap old and new worlds
        tmp_world = cur_world;
        cur_world = next_world;
        next_world = tmp_world;
        
        if (print_cells > 0 || print_world > 0)
        {
            // send updated parts back to process 0
	        gather_updated_parts(&whole_world, cur_world, processor_rank, 
                                   number_of_processors, widths, starts);
            if (print_cells > 0 && (n % print_cells) == (print_cells - 1) 
                                && (processor_rank == 0)) {
                printf("%d: %d live cells\n", n, world_count(&whole_world));
            }

            if (print_world > 0 && (n % print_world) == (print_world - 1) 
                                && (processor_rank == 0)) {
                printf("\nafter time step %d - rank %d:\n\n", n, processor_rank);
                world_print(&whole_world);
            }
        }
    }
    // stop time
    end_time = time_secs();
    elapsed_time = end_time - start_time;
    // get the elapsed time of the slowest process
    double slowest_process_time;
    MPI_Reduce(&elapsed_time, &slowest_process_time, 1, MPI_DOUBLE, 
					            MPI_MAX, 0, MPI_COMM_WORLD);
    
    /* iterations are done; sum the number of live cells */
    int global_sum;
    int local_sum = world_count(cur_world);
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (processor_rank == 0)
    {
    	printf("Number of live cells = %d\n", global_sum);
    	fprintf(stderr, "Game of Life took %10.3f seconds\n", elapsed_time);
    }
   
    MPI_Finalize();
    
    return 0;
}
