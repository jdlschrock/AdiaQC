#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include "support/parson.h"
#include <time.h> //for timing purposes

typedef struct {
    uint64_t nQ;
    uint64_t dim;
    uint64_t L;
    double dt;
    double T;
    double *al;
    double *be;
    double *de;
} params_t;

inline double A( double t ){
    return 1.0 - t;
}
inline double B( double t ){
    return t;
}

typedef struct {
    uint64_t ldim;
    ptrdiff_t N0, N1, n0, n1;
    ptrdiff_t alloc_local, local_N0, local_0_start;
    ptrdiff_t local_N1, local_1_start;
} grid_t;

typedef struct {
    unsigned long size,resident,share,text,lib,data,dt;
} statm_t;

void read_off_memory_status(statm_t* result){
    unsigned long dummy;
    const char* statm_path = "/proc/self/statm";
    FILE *f = fopen(statm_path,"r");  
    
    if(!f){
        fprintf(stderr, statm_path);
        abort();
    }
    
    if(7 != fscanf(f,"%ld %ld %ld %ld %ld %ld %ld",
                    &result->size,&result->resident,
                    &result->share,&result->text,&result->lib,
                    &result->data,&result->dt)){
        fprintf(stderr, statm_path);
        abort();
    }
    
    fclose(f);
}

//TODO: going to need logic to handle incomplete config files
void parse_file( char *file, params_t *par ){
    /* Parse file and populate applicable data structures */
    uint64_t i;
    JSON_Value *root_value = NULL;
    JSON_Object *root_object;
    JSON_Array *array;

    root_value = json_parse_file_with_comments( file );
    root_object = json_value_get_object( root_value );

    par->nQ = (uint64_t) json_object_dotget_number( root_object, "scalars.nq" );
    par->L = (uint64_t) json_object_dotget_number( root_object, "scalars.lrgs" );
    par->T = json_object_dotget_number( root_object, "scalars.t" );
    par->dt = json_object_dotget_number( root_object, "scalars.dt" );

    par->al   = (double *)malloc( (par->nQ)*sizeof(double) );
    par->de   = (double *)malloc( (par->nQ)*sizeof(double) );
    par->be   = (double *)malloc( ((par->nQ)*((par->nQ)-1)/2)*sizeof(double) );

    array = json_object_dotget_array( root_object, "coefficients.alpha" );
    if( array != NULL ){
        for( i = 0; i < json_array_get_count(array); i++ ){
            (par->al)[i] = -json_array_get_number( array, i );
        }
    }

    array = json_object_dotget_array( root_object, "coefficients.beta" );
    if( array != NULL ){
        for( i = 0; i < json_array_get_count(array); i++ ){
            (par->be)[i] = -json_array_get_number( array, i );
        }
    }

    array = json_object_dotget_array( root_object, "coefficients.delta" );
    if( array != NULL ){
        for( i = 0; i < json_array_get_count(array); i++ ){
            (par->de)[i] = -json_array_get_number( array, i );
        }
    }

    json_value_free( root_value );
}

void build_h( uint64_t dim, uint64_t base, const params_t *par, double *hz, double *hhxh, double complex *psi ){
    uint64_t i, j, k, testi, testj, bcount;
    int dzj, dzi;
    double complex factor = 1.0/sqrt( par->dim );

    //Assemble Hamiltonian and state vector
    for( k = 0; k < gr->ldim; k++ ){
        bcount = 0;
        for( i = 0; i < (par->nQ); i++ ){
            testi = 1 << ((par->nQ) - i - 1);
            dzi = ( ( (k + base)/testi ) & 1 ) ? 1 : -1;

            hz[k] += (par->al)[i] * dzi;
            hhxh[k] += (par->de)[i] * dzi;

            for( j = i+1; j < (par->nQ); j++ ){
                testj = 1 << ((par->nQ) - j - 1);
                dzj = ( ( (k + base)/testj) & 1 ) ? 1 : -1;

                hz[k] += (par->be)[bcount] * dzi * dzj;
                bcount++;
            }
        }
            
        psi[k] = factor;
    }
}

void expMatTimesVec( uint64_t N, double complex cc, double complex scale, const double *mat, double complex *vec ){
    uint64_t i;

    for( i = 0UL; i < N; i++ ){
        vec[i] *= scale * cexp( cc*mat[i] );
    }
}

/* BEGIN FWHT METHODS */
void FWHT( uint64_t nQ, uint64_t dim, double complex *vec ){
    uint64_t stride, base, j;
    
    //Cycle through stages with different butterfly strides
    for( stride = dim / 2; stride >= 1; stride >>= 1 ){   
        //Butterfly index within subvector of (2 * stride) size
        for( j = 0; j < dim/2; j++ ){   
            base = j - (j & (stride-1));
            
            uint64_t i0 = base + j +      0;  
            uint64_t i1 = base + j + stride;
            
            double complex T1 = vec[i0];
            double complex T2 = vec[i1];
            vec[i0] = T1 + T2; 
            vec[i1] = T1 - T2; 
        }
    }   
}

//void FWHTP( fftw_complex *vec, ptrdiff_t N0, ptrdiff_t N1, ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t local_n0, ptrdiff_t local_n1 ){
void FWHTP( const grid_t *gr, fftw_plan planT, fftw_plan planU, double complex *psi ){
    uint64_t j;

    for( j = 0UL; j < gr->local_n0; ++j){
        FWHT( n1, gr->N1, &vec[j*(gr->N1)] );
    }
    fftw_execute( planT );

    for( j = 0UL; j < gr->local_n1; ++j){
        FWHT( n0, gr->N0, &vec[j*(gr->N0)] );
    }
    fftw_execute( planU );
}
/* END FWHT METHODS */
   
//TODO: finish this
void run_sim( const params_t *par, const grid_t *gr, const double *hz, const double *hhxh, double complex *psi ){
    uint64_t i, N = (uint64_t)((par->T) / (par->dt));
    double complex cz, cx, im=(-(par->dt)*I);
    double t;

    for( i = 0; i <= N; ++i ){
        t = i*(par->dt)/(par->T);

        //Time-dependent coefficients
        cz = (im/2.0) * B(t);
        cx = im * A(t);

        //Evolve system
        expMatTimesVec( ldim, cz, 1.0, hz, psi ); //apply Z part
        FWHTP( gr, planT, planU, psi ); //transpose
        expMatTimesVec( ldim, cx, 1.0, hhxh, psi ); //apply X part
        FWHTP( gr, planT, planU, psi ); //transpose
        expMatTimesVec( ldim, cz, 1.0/dim, hz, psi ); //apply Z part
    }
    fftw_destroy_plan( planT ); //transpose
    fftw_destroy_plan( planU ); //transpose
}

//from: http://www.dreamincode.net/forums/topic/61496-find-n-max-elements-in-unsorted-array/
//author: baavgai
//date: 2014-09-08, 16:30
void addLarger( uint64_t indx, uint64_t size, double value, uint64_t *list, double *mag_list ){
    uint64_t i = 0;
    while( i < size-1 && value > mag_list[i+1] ){
        mag_list[i] = mag_list[i+1];
        list[i] = list[i+1];
        i++;
    }
    list[i] = indx;
    mag_list[i] = value;
}

void findLargest( uint64_t size, uint64_t N, const double complex *list, uint64_t *listN ){
    uint64_t i;
    double temp;
    
    double *mag_list = (double *)calloc( N, sizeof(double) );

    for( i = 0; i < N; i++ ){
        addLarger( i, N, cabs( list[i] ), listN, mag_list );
    }
    for( i = N; i < size; i++ ){
        temp = cabs( list[i] );
        if( temp > *mag_list ){
            addLarger( i, N, temp, listN, mag_list );
        }
    }
    free( mag_list );
}

int main( int argc, char **argv ){
    //TODO: compare vars to the bare min in simulator.c
    //      everything else in prob_t
    double *hz, *hhxh;     /* hamiltonian components */
    double complex *psi; //*work;   /* State vector */
    params_t par;
    uint64_t i, *large, *largest;
    clock_t begin, end;
    MPI_Status status;
    int id, np;
    grid_t gr;
    double complex *clarge, *clargest;
    fftw_plan planT, planU; //transpose

    MPI_Init( &argc, &argv );
    fftw_mpi_init();
     
    MPI_Comm_rank( MPI_COMM_WORLD, &id );
    MPI_Comm_size( MPI_COMM_WORLD, &np );
    
    if( id == 0 ){
        begin = clock(); //TIMING
    }

    if( id == 0 ){
        if( argc < 2 ){
            fprintf( stderr, "Need a json configuration file. Terminating...\n" );
            return 1;
        }
        parse_file( argv[1], &par );
    }
    MPI_Bcast( &par.nQ, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD );
    MPI_Bcast( &par.L, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD );
    MPI_Bcast( &par.t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Bcast( &par.dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    if( id != 0 ){
        par.al   = (double *)malloc( par.nQ*sizeof(double) );
        par.de   = (double *)malloc( par.nQ*sizeof(double) );
        par.be   = (double *)malloc( (par.nQ*(par.nQ-1)/2)*sizeof(double) );
    }
    MPI_Bcast( par.al, par.nQ, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Bcast( par.de, par.nQ, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Bcast( par.be, (par.nQ*(par.nQ-1)/2), MPI_DOUBLE, 0, MPI_COMM_WORLD );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Compute the Hamiltonian and state vector for the simulation
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
        Create state vector and initialize to 1/sqrt(2^n)*(|00...0> + ... + |11...1>)
    */
    par.dim = 1UL << par.nQ;
    gr.n0 = par.nQ/2;
    gr.n1 = par.nQ - gr.n0;
    gr.N0 = 1UL << gr.n0; //this should be the number of processors?
    gr.N1 = 1UL << gr.n1; //also defines the order of the WHT product
    
    gr.alloc_local = fftw_mpi_local_size_2d_transposed( gr.N0, gr.N1, MPI_COMM_WORLD,
                                                     &gr.local_N0, &gr.local_0_start,
                                                     &gr.local_N1, &gr.local_1_start );
    gr.local_0_start *= gr.N1; 
    psi = fftw_alloc_complex( gr.alloc_local );
    gr.ldim = gr.local_N0 * gr.N1;
    hz   = (double *)calloc( gr.ldim, sizeof(double) );
    hhxh = (double *)calloc( gr.ldim, sizeof(double) );

    /*
        Assemble Hamiltonian and state vector
    */
    build_h( gr.ldim, gr.local_0_start, &par, hz, hhxh, psi );
    free( par.al ); free( par.be ); free( par.de );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                            Run the Simulation
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    //TODO: put these into grid_t and turn it into prob_t?
    planT = fftw_mpi_plan_many_transpose( gr.N0, gr.N1, 2, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
            (double *)psi, (double *)psi, MPI_COMM_WORLD, FFTW_ESTIMATE ); //transpose
    planU = fftw_mpi_plan_many_transpose( gr.N1, gr.N0, 2, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
            (double *)psi, (double *)psi, MPI_COMM_WORLD, FFTW_ESTIMATE ); //transpose

    run_sim( &par, &gr, planT, planU, hz, hhxh, psi );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Check solution and clean up
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    large = (uint64_t *)calloc( par.L, sizeof(uint64_t) );
    clarge = (double complex *)calloc( par.L, sizeof(double complex) );
    if( id == 0 ){
        largest = (uint64_t *)malloc( np*sc.L*sizeof(uint64_t) );
        clargest = (double complex *)malloc( np*sc.L*sizeof(double complex) );
    }

    findLargest( ldim, par.L, psi, large );
    for( i = 0; i < par.L; ++i ){
        clarge[i] = psi[ large[i] ];
        large[i] = local_0_start + i;
    }

    MPI_Gather( clarge, par.L, MPI_C_DOUBLE_COMPLEX, clargest, par.L, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD );
    MPI_Gather( large, par.L, MPI_UNSIGNED_LONG, largest, par.L, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD );
    if( id == 0 ){
        findLargest( np*par.L, par.L, clargest, large ); //zero out large?
        //for( i = 0; i < par.L; ++i ){
        for( i = par.L - 1; i >= 0; --i ){
            printf( "|psi[%llu]| = %f\n",
                largest[ large[i] ],
                cabs( clargest[large[i]]*clargest[large[i]] ) );
        }
        statm_t res;
        read_off_memory_status( &res );
        end = clock();
        printf( "Total time: %f s\n", (double)(end - begin)/CLOCKS_PER_SEC );
        printf( "Memory used: %ld kB\n", res.resident );
    }

    free( large );
    free( clarge );
    if( id == 0 ){
        free( largest );
        free( clargest );
    }

    fftw_free( psi );
    free( hz ); free( hhxh );
    MPI_Finalize();

    return 0;
}
