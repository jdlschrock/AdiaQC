#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
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

//TODO: figure out this with mpi
void build_h( const params_t *par, double *hz, double *hhxh, double complex *psi ){
    uint64_t i, j, k, testi, testj, bcount;
    //uint64_t dim = 1 << (par->nQ);
    int dzj, dzi;
    double complex factor = 1.0/sqrt( par->dim );

    //Assemble Hamiltonian and state vector
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
            
        psi[k] = factor;
    }
}

void expMatTimesVec( uint64_t N, double complex cc, double complex scale, const double *mat, double complex *vec ){
    uint64_t i;

    for( i = 0UL; i < N; i++ ){
        vec[i] *= scale * cexp( cc*mat[i] );
    }
}

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

void run_sim( const params_t *par, const double *hz, const double *hhxh, double complex *psi ){
    uint64_t i, N;//, dim = 1 << (par->nQ);
    double complex cz, cx, im=(-(par->dt)*I), fac=1.0/(par->dim);
    double t;
    N = (uint64_t)(par->T / par->dt);

    for( i = 0; i <= N; ++i ){
        t = i*(par->dt)/(par->T);

        //Time-dependent coefficients
        cz = (im/2.0) * B(t);
        cx = im * A(t);

        //Evolve system
        expMatTimesVec( par->dim, cz, 1.0, hz, psi ); //apply Z part
        FWHT( par->nQ, par->dim, psi );
        expMatTimesVec( par->dim, cx, 1.0, hhxh, psi ); //apply X part
        FWHT( par->nQ, par->dim, psi );
        expMatTimesVec( par->dim, cz, fac, hz, psi ); //apply Z part
    }
}

//from: http://www.dreamincode.net/forums/topic/61496-find-n-max-elements-in-unsorted-array/
//author: baavgai
//date: 2014-09-08, 1630
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
    uint64_t i;
    struct timeval tend, tbegin;
    double delta;
    
    gettimeofday( &tbegin, NULL );
    
    /* - - - - - - - - - - Parse configuration file - - - - - - - - - - - -*/
    if( argc < 2 ){
        fprintf( stderr, "Need a json configuration file. Terminating...\n" );
        return 1;
    }
    
    parse_file( argv[1], &par );
    
    /* - - Compute the Hamiltonian and state vector for the simulation - - */
    par.dim = 1 << par.nQ;
    
    psi  = (double complex *)malloc( (par.dim)*sizeof(double complex) );
    hz   = (double *)calloc( (par.dim),sizeof(double) );
    hhxh = (double *)calloc( (par.dim),sizeof(double) );
    
    build_h( &par, hz, hhxh, psi );
    free( par.al ); free( par.be ); free( par.de );
    
    /* - - - - - - - - - - - Run the Simulation - - - - - - - - - - - - - -*/
    run_sim( &par, hz, hhxh, psi );
    
    /* - - - - - - - - - - - - - Check results - - - - - - - - - - - - - - */
    uint64_t *largest = (uint64_t *)calloc( par.L, sizeof(uint64_t) );
    findLargest( par.dim, par.L, psi, largest );
    for( i = par.L; i --> 0; ){
        printf( "|psi[%llu]| = %.8f\n",
            largest[i],
            cabs( psi[largest[i]]*psi[largest[i]] ) );
    }
    statm_t res;
    read_off_memory_status( &res );
    free( largest );
    
    //Free work space
    free( psi );
    free( hz );
    free( hhxh );
    
    //Print system information
    gettimeofday( &tend, NULL );
    delta = ((tend.tv_sec - tbegin.tv_sec)*1000000u + tend.tv_usec - tbegin.tv_usec)/1.e6;
    printf( "Total time: %f s\n", delta );
    printf( "Memory used: %ld kB\n", res.resident );
    
    return 0;
}
