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

/* each process decides which are the top and bottom
   neighbouring processes to communicate boundary conditions */
void
decide_neighbours(int* top, int* bottom, int rank, int p)
{
    // process 0 sends top row to last process
    // and bottom row to process next process
    if (rank == 0) {
	    *top = p - 1;
        if (p == 1) { // in case of a single process neigbours are myself
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
    int i;

    /* left-right boundary conditions */
    for (i = 1; i <= world->height; i++) {
        cells[i][0] = cells[i][world->width];
        cells[i][world->width + 1] = cells[i][1];
    }
    
    /* top-bottom boundary conditions */
    // get the neighbouring processes
    int top_neighbour, bottom_neighbour;
    decide_neighbours(&top_neighbour, &bottom_neighbour, rank, p); 
    
    // communicate rows with neighbours
    if (rank % 2 == 0) // even ranks send first then receive
    {   // send first real (not ghost) row to top neighbour
        MPI_Send(&cells[1][0], world->width + 2, MPI_INT, 
                       top_neighbour, 2, MPI_COMM_WORLD);
	    // receive bottom neighbour's first real row and set it as bottom ghost
        MPI_Recv(&cells[world->height + 1][0], world->width + 2, MPI_INT, 
                 bottom_neighbour, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    // send last real row to bottom neighbour
        MPI_Send(&cells[world->height][0], world->width + 2, MPI_INT,
                    	        bottom_neighbour, 2, MPI_COMM_WORLD);
	    // receive top neighbour's last real row and set it as top ghost
        MPI_Recv(&cells[0][0], world->width + 2, MPI_INT,
                 top_neighbour, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else { // odd ranks receive first then send
        MPI_Recv(&cells[world->height + 1][0], world->width + 2, MPI_INT, 
                 bottom_neighbour, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&cells[1][0], world->width + 2, MPI_INT, 
                       top_neighbour, 2, MPI_COMM_WORLD);
        MPI_Recv(&cells[0][0], world->width + 2, MPI_INT,
                 top_neighbour, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&cells[world->height][0], world->width + 2, MPI_INT,
                                bottom_neighbour, 2, MPI_COMM_WORLD);
    }
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
world_timestep(world *old, world *new, int rank, int p)
{
    int i, j;
    
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


/* decide the number of rows and the starting row of each process */
void
grid_decomposition(int height, int rank, int p, 
                   int* subworld_start, int* subworld_height) 
{
	*subworld_start = height / p * rank;
	*subworld_height = height / p;
	int remainder = height % p;
	// give 1 more row to processes of rank < remainder
	if (rank < remainder) {
		*subworld_height += 1;
	}
	// adjust the processes starting rows
	if (rank > remainder) {
		*subworld_start += remainder;
	} else {
		*subworld_start += rank;
	}		
}

/* split the world to subworlds and scatter them to processes */
void
split_board(world *whole_world, world *subworld, 
            int p, int rank, int start_row, int *heights, int *starts)
{
    // initialize Recv buffer
    int receive[2 * (p - 1)];
    // process 0
    if (rank == 0)
    {	
        heights[0] = subworld->height * (subworld->width + 2);
        starts[0] = start_row * (subworld->width + 2);
        
	    for (int n = 1; n < p; n++)
        {   
            // receive from each process their starting row 
            // and the number of rows they will work on
            // (results of grid_decomposition)
            MPI_Recv(&receive[2 * (n - 1)], 2, MPI_INT, 
                     n, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // put the received info in their corresponding array (heights and starts)
        // to prepare them for MPI_Scatterv and later MPI_Gatherv
        for (int i = 0; i < 2 * (p - 1); i++)
        {
            if (i % 2 == 0) {
                starts[i / 2 + 1] = receive[i] * (subworld->width + 2);
            } else {
                heights[(int)floor(i / 2) + 1] = receive[i] * (subworld->width + 2);
            }
        }
        // send to each process its part of the whole_world
        // (without top-bottom ghost cells)
        MPI_Scatterv(&whole_world->cells[1][0], heights, starts, MPI_INT, 
                     &cur_world->cells[1][0], heights[0], MPI_INT, 0, MPI_COMM_WORLD);
    // other processes
    } else {
        // send my subworld_start and subworld_height to process 0
        int send[2] = {start_row, subworld->height};
        MPI_Send(&send, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
        // reveive my part of the whole_world, 
        // and save it to my current world
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, &cur_world->cells[1][0], 
                     subworld->height * (subworld->width + 2), MPI_INT, 0, MPI_COMM_WORLD);
    }
}

/* gather the updated subworlds from all processes to process 0 */
void
gather_updated_parts(world *whole_world, world *subworld,
                           int rank, int p, int *heights, int *starts)
{
    // process 0 reassembles updated world for printing purposes
    if (rank == 0)
    {	
        MPI_Gatherv(&subworld->cells[1][0], heights[0],
                    MPI_INT, &whole_world->cells[1][0], 
                    heights, starts, MPI_INT, 0, MPI_COMM_WORLD);
    // other processes send their updated parts to process 0
    // (without top-bottom ghost cells)
    } else {
        MPI_Gatherv(&cur_world->cells[1][0],
                    subworld->height * (subworld->width + 2), MPI_INT, 
                    NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }
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

    if (bheight < number_of_processors) {
        if (processor_rank == 0) {
            fprintf(stderr, "The world height must be equal to or greater than the number of processes.\n");
        }
        exit(1);
    }

    // decide the rows each process will work on
    // variables for starting row and number of rows of each process
    int subworld_start, subworld_height;
    grid_decomposition(bheight, processor_rank, number_of_processors,
			                      &subworld_start, &subworld_height);
    
    /* initialize worlds, when allocating arrays, add 2 for ghost cells in both directorions */
    // only their part of the grid is allocated in each process to save memory
    worlds[0].height = subworld_height;
    worlds[0].width = bwidth;
    worlds[0].cells = alloc_2d_int_array(subworld_height + 2, bwidth + 2);

    worlds[1].height = subworld_height;
    worlds[1].width = bwidth;
    worlds[1].cells = alloc_2d_int_array(subworld_height + 2, bwidth + 2);

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
    
    // initialize a random or fixed board in process 0 
    if (processor_rank == 0)
    {
        if (random_world) {
            world_init_random(&whole_world);
        } else {
            world_init_fixed(&whole_world);
        }
    }

    // arrays to save the number of rows and starting row of each process
    // (process 0 needs this information for MPI_Scatterv and MPI_Gatherv)
    int heights[number_of_processors], starts[number_of_processors];
    
    // send to each process its part
    split_board(&whole_world, cur_world, number_of_processors, 
                          processor_rank, subworld_start, heights, starts);
    
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
        world_timestep(cur_world, next_world, processor_rank,
						     number_of_processors);

        // swap old and new worlds
        tmp_world = cur_world;
        cur_world = next_world;
        next_world = tmp_world;
        
        if (print_cells > 0 || print_world > 0)
        {
            // send updated parts back to process 0
	        gather_updated_parts(&whole_world, cur_world, processor_rank, 
                                  number_of_processors, heights, starts);
	        // print number of alive cells at this step
            if (print_cells > 0 && (n % print_cells) == (print_cells - 1) 
                                && (processor_rank == 0)) {
                printf("%d: %d live cells\n", n, world_count(&whole_world));
            }
	        // print the whole world at this step
            if (print_world > 0 && (n % print_world) == (print_world - 1) 
                                && (processor_rank == 0)) {
                printf("\nafter time step %d - rank %d:\n\n", n, processor_rank);
                world_print(&whole_world);
            }
        }
    }
    // stop measuring time
    end_time = time_secs();
    elapsed_time = end_time - start_time;
    // get the elapsed time of the slowest process
    double slowest_process_time;
    MPI_Reduce(&elapsed_time, &slowest_process_time, 1,
	                                MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    /* Iterations are done; sum the number of live cells */
    int global_sum;
    int local_sum = world_count(cur_world);
    // send and add all local sums
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (processor_rank == 0)
    {
    	printf("Number of live cells = %d\n", global_sum);
    	fprintf(stderr, "Game of Life took %10.3f seconds\n", slowest_process_time);
    }
    
    MPI_Finalize();
    
    return 0;
}
