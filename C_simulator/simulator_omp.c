#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include "support/parson.h"
#include <omp.h>
#include <time.h> //for timing purposes

typedef struct {
    uint64_t nQ;
    uint64_t dim;
    uint64_t L;
    uint64_t res;
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
void parse_file( char type, char *json, params_t *par ){
    /* Parse file and populate applicable data structures */
    uint64_t i;
    JSON_Value *root_value = NULL;
    JSON_Object *root_object;
    JSON_Array *array;

    if( type == 'f' )
        root_value = json_parse_file_with_comments( json );
    else
        root_value = json_parse_string_with_comments( json );

    root_object = json_value_get_object( root_value );

    par->nQ = (uint64_t) json_object_dotget_number( root_object, "scalars.nq" );
    par->L = (uint64_t) json_object_dotget_number( root_object, "scalars.lrgs" );
    par->res = (uint64_t) json_object_dotget_number( root_object, "scalars.res" );
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

void build_h( const params_t *par, double *hz, double *hhxh ){
    uint64_t i, j, k, testi, testj, bcount;
    int dzj, dzi;

    //Assemble Hamiltonian and state vector
    #pragma omp parallel for private(i, j, bcount, testi, testj, dzi, dzj)
    for( k = 0; k < par->dim; k++ ){
        bcount = 0;
        for( i = 0; i < (par->nQ); i++ ){
            testi = 1 << ((par->nQ) - i - 1);
            dzi = ( (k/testi) & 1 ) ? 1 : -1;

            hz[k] += (par->al)[i] * dzi;
            hhxh[k] += (par->de)[i] * dzi;

            for( j = i+1; j < (par->nQ); j++ ){
                testj = 1 << ((par->nQ) - j - 1);
                dzj = ( (k/testj) & 1 ) ? 1 : -1;

                hz[k] += (par->be)[bcount] * dzi * dzj;
                bcount++;
            }
        }
    }
}

void init_psi( const params_t *par, double complex *psi ){
    uint64_t k;
    double sum=0.0;
    double complex scale;
    //double complex factor = 1.0/sqrt( par->dim );    
    
    srand( time(NULL) );
    #pragma omp parallel for reduction(+:sum)
    for( k = 0; k < par->dim; ++k){
        //psi[k] = factor;
        //psi[k] = ((double complex)rand())/RAND_MAX;
        psi[k] = (double complex)(rand() & 1);
        sum += creal(psi[k]); //normalize state vector
    }
    
    //Normalize the state vector
    scale = (double complex)sqrt( sum );
    #pragma omp parallel for
    for( k = 0; k < par->dim; ++k ){
        psi[k] = psi[k]/scale;;
    }
}

void expMatTimesVec( uint64_t N, double complex cc, double complex scale, const double *mat, double complex *vec ){
    uint64_t i;

    #pragma omp parallel for
    for( i = 0UL; i < N; i++ ){
        vec[i] *= scale * cexp( cc*mat[i] );
    }
}

void FWHT( uint64_t nQ, uint64_t dim, double complex *vec ){
    uint64_t stride, base, j;
    
    //Cycle through stages with different butterfly strides
    for( stride = dim / 2; stride >= 1; stride >>= 1 ){   
        //Butterfly index within subvector of (2 * stride) size
        #pragma omp parallel for private(base)
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

void run_sim( const params_t *par, const double *hz, const double *hhxh, double complex *psi ){
    uint64_t i, N;//, dim = 1 << (par->nQ);
    double complex cz, cx, im=(-(par->dt)*I), fac=1.0/(par->dim);
    double t0, t1;
    N = (uint64_t)((par->T) / (par->dt));

    cz = ( im/2.0 )*B(0);
    expMatTimesVec( par->dim, cz, 1.0, hz, psi ); //apply Z part
    for( i = 0; i <= N; ++i ){
        t0 = i*(par->dt)/(par->T);
        t1 = (i+1)*(par->dt)/(par->T);

        //Time-dependent coefficients
        cz = ( im/2.0 )*( B(t0) + ((i < N) ? B(t1) : 0) ); 
        cx = im * A(t0);

        //Evolve system
        FWHT( par->nQ, par->dim, psi );
        expMatTimesVec( par->dim, cx, 1.0, hhxh, psi ); //apply X part
        FWHT( par->nQ, par->dim, psi );
        expMatTimesVec( par->dim, cz, fac, hz, psi ); //apply Z part
    }
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

/*------------------------------------------------------------------
Main driver for the simulator
------------------------------------------------------------------*/
int main( int argc, char **argv ){
    double *hz, *hhxh;     /* hamiltonian components */
    double complex *psi;   /* State vector */
    params_t par;
    uint64_t i, *largest, j, samples, maxIdx, ccount;
    struct timeval tend, tbegin;
    double delta;
    double tempV, maxV;
    
    //gettimeofday( &tbegin, NULL );
    
    /* - - - - - - - - - - - Parse configuration file - - - - - - - - - - -*/
    //if( argc < 3 ){
    if( argc < 4 ){
        fprintf( stderr, "Need a json configuration file or json string. Terminating...\n" );
        return 1;
    }

    parse_file( argv[1][1], argv[2], &par );
    samples = atoi( argv[3] );
    
    par.dim = 1 << par.nQ;
    
    hz   = (double *)calloc( (par.dim),sizeof(double) );
    hhxh = (double *)calloc( (par.dim),sizeof(double) );
    psi  = (double complex *)malloc( (par.dim)*sizeof(double complex) );
    
    /* - - - - - - Compute the Hamiltonian for the simulation - - - - - - -*/
    build_h( &par, hz, hhxh );
    free( par.al ); free( par.be ); free( par.de );

    ccount = 0UL;
    for( i = 0; i < samples; ++i ){
        /* - - - - - - Compute the state vector for the simulation - - - - - - */
        init_psi( &par, psi );
        
        /* - - - - - - - - - - - - Run the Simulation - - - - - - - - - - - - -*/
        run_sim( &par, hz, hhxh, psi );

        /* - - - - - - - - - - - - - Check results - - - - - - - - - - - - - - */
        maxV = 0.0;
        for( j = 0UL; j < par.dim; ++j ){
            tempV = cabs( psi[j]*psi[j] );
            if( tempV > maxV ){
                maxV = tempV;
                maxIdx = j;
            }
        }
        ccount += ( maxIdx == par.res ) ? 1UL : 0UL;
    }
    printf( "%f\n", ccount/(double)samples );

    /* - - - - - - - - - - - - - Check results - - - - - - - - - - - - - - */
    /*
    largest = (uint64_t *)calloc( par.L, sizeof(uint64_t) );
    findLargest( par.dim, par.L, psi, largest );
    for( i = par.L; i --> 0; ){ //remember that i is unsigned
        //printf( "|psi[%d]| = %.8f\n",
        printf( "%d %.8f\n",
            largest[i],
            cabs( psi[largest[i]]*psi[largest[i]] ) );
    }
    */
    //statm_t res;
    //read_off_memory_status( &res );
    
    /* - - - - - - - - - - - Clean up and output - - - - - - - - - - - - - */
    //free( largest );
    free( psi );
    free( hz );
    free( hhxh );
    
    //gettimeofday( &tend, NULL );
    //delta = ((tend.tv_sec - tbegin.tv_sec)*1000000u + tend.tv_usec - tbegin.tv_usec)/1.e6;
    //printf( "Total time: %f s\n", delta );
    //printf( "Memory used: %ld kB\n", res.resident );
    
    return 0;
}
