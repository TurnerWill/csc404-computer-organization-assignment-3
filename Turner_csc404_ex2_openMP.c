/***********************************************
* William Turner CSC 404					   *
* Assignment 3 benchmarking matrix multiply    *
*			   with openMP parallelization 	   *
***********************************************/

#include<time.h> /* for random number generator (filling of matrix column/row array elements),
			    and clock functions */
#include<stdio.h>
#include<stdlib.h> /* for malloc (memory allocation) of dynamic arrays*/
#include<math.h>
#include<omp.h> 

#define N 128 //dimension for rows and columns of square-matrices
#define LOOPS 1 // number of accuracy improvement loops
#define CPU_CLK 2.0e9 // toshiba satellite laptop AMD A6-5200 quad-core

#define THREAD_COUNT 4

// ((2*pow(N,3)) - pow(N,2)); formula used to determine number of matrix arithmetic operations

const int ROW_SIZE = N;
const int COL_SIZE = N;

void printTitle();
void printArray(double **a); //print to screen function for testing/debugging
void deallocate(double **a, int z); // memory deallocation

int main()
{
	int a, b, c, i, j, k, l, p, q; // for-loop counters
	double t_id, t_id2; // for openMP time functions
	double ** matrixA;
	double ** matrixB;
	double ** matrixC;
	double arithmetic_ops = ((2*pow(N,3)) - pow(N,2)); // # of arithmetic ops in matrix multiply

	srand(time(NULL));

	// allocate memory for matrix A rows
	matrixA = malloc(ROW_SIZE*sizeof(double*));
	//allocate memory for matrix A columns
	for(a=0; a<ROW_SIZE;a++)
	{
		matrixA[a] = malloc(COL_SIZE*sizeof(double*));
	}

	//fill matrixA multidimensional array with random numbers
	for(i=0; i<ROW_SIZE; i++)
	{	
		for(j=0; j<COL_SIZE; j++)
		{
			matrixA[i][j] = rand() % 100;
		
		}
	}
		
	// allocate memory for matrix B rows	
	matrixB = malloc(ROW_SIZE*sizeof(double*));
	//allocate memory for matrix B columns
	for(b=0; b<ROW_SIZE;b++)
	{
		matrixB[b] = malloc(COL_SIZE*sizeof(double*));
	}	
	//fill matrixB multidimensional array with random numbers
	for(k=0; k<ROW_SIZE; k++)
	{	
		for(l=0; l<COL_SIZE; l++)
		{
			matrixB[k][l] = rand() % 100;
		
		}
	}
		
	// allocate memory for matrix C rows	
	matrixC = malloc(ROW_SIZE*sizeof(double*));
	//allocate memory for matrix C columns
	for(c=0; c<COL_SIZE;c++)
	{
		matrixC[c] = malloc(COL_SIZE*sizeof(double*));
	}
	// fill matrixC with 0s (clear all junk values for forthcoming additions and multiplications)
	for(p=0; p<ROW_SIZE; p++)
	{	
		for(q=0; q<COL_SIZE; q++)
		{
			matrixC[p][q] = 0;		
		}
	}

	printTitle();
	printf("performing matrix multiply operations...");

	/***************************************
 	timed portion with openMD parallelization
 	***************************************/	
	t_id = omp_get_wtime();

	int x, s, t, v; //loop counters for omp parallelized matrix multiply section
	
	#pragma omp parallel num_threads(THREAD_COUNT) 				
	for(x=0; x<LOOPS; x++) //accuracy loop
	{
		// fill matrixC with results of matrixA * matrixB (matrix multiplication) using openMP
		// thread assignment in for loop
		#pragma omp for private(s,t,v)
		for( s=0; s<ROW_SIZE; s++)
		{	
			for( t = 0; t<ROW_SIZE; t++)
			{
				for( v = 0; v < COL_SIZE; v++)
				{
					matrixC[s][t] += matrixA[s][v] * matrixB[v][t];
				}	
			}			
		} 	 
	} //end accuracy loop
	
	t_id2 = omp_get_wtime();

	system("cls");					

	//clean up memory allocation
	deallocate(matrixA, ROW_SIZE);
	deallocate(matrixB, ROW_SIZE);
	deallocate(matrixC, ROW_SIZE);

	printTitle();

	printf("****************\n");
	printf("*    Results   *\n");
	printf("****************\n\n");
	printf("# of threads opened through openMP:\t %d\n\n", THREAD_COUNT);
	printf("matrices dimension:\t\t\t %d X %d\n",N,N);
	printf("# of elements per matrix:\t\t %d\n", N*N);
	printf("# of accuracy improvement loops:\t %d\n", LOOPS);
	printf("number of arithmetic operations:\t %.2lf\n", arithmetic_ops);	
	printf("\nTotal Computation Time:\t\t\t %.3e seconds\n", (double)((t_id2-t_id) * 1000)/CLOCKS_PER_SEC);
	printf("\t\t\t\t\t (includes accuracy loop times)\n");
	double matrix_time = ((double)((t_id2-t_id) * 1000)/CLOCKS_PER_SEC)/LOOPS;
	printf("\nNxN Matrix Multiply Computation Time:\t %.2lf seconds", matrix_time);
	printf(" (C[] = A[] * B[])\n");
	double per_element_time = matrix_time / (N*N);
	printf("\nComputation Time\nper Arithmetic Operation:\t\t %.3e\n", matrix_time / arithmetic_ops);
	printf("\nNumber of Cycles\nper Arithmetic Operation:\t\t %.4f\n", CPU_CLK / (arithmetic_ops / matrix_time));

	return 0;
 
} // end main

/**************************************************************
* prints matrix to screen for testing/debugging purposes only *
**************************************************************/
void printArray(double** a)
{
	int m, n;
	for (m=0; m<ROW_SIZE;m++ )
	{
		for (n = 0; n <= COL_SIZE-1; n++)
		{
			printf("%.2lf ", a[m][n]);
		}
	
		printf("\n");
	}
}

/*********************************
* 		memory deallocation		 *
*********************************/	
void deallocate(double **a, int z)
{
	int x;
	for ( x = 0; x < z; x++)
	{
		free(a[x]);
	}
	free(a);
}

/*******************************
* print title					*
********************************/
void printTitle()
{
	printf("William Turner\n");
	printf("CSC404	Programming Exercise #3\n");
	printf("Matrix Multiplication Benchmarking with openMP thread parallelization\n\n");
}
