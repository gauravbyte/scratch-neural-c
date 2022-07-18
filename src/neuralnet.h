#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<string.h>
#include <string.h>


typedef struct
{
    double *list_weight, *prev_list_weight ;
    int size ;

} Neuron ;

typedef struct
{
    int neuron_count ;
    Neuron *neuraon_list ;
} Layer ;

typedef struct
{
    int layer_count , feature_count ;
    Layer *arr_layer ;
} Network ;


double * forward_layer(Network *nn,double *input_layer,int layer,int act) ;
double mean_squared_err(Network *nn,double **matrix_input,double **output_layer,int train_data) ;
double cross_entropy_loss(Network *nn,double **matrix_input,double **output_layer,int train_data) ;
double * calculate_delta(Network *nn, int layer, double *input_layer, double *output_layer_single) ;
void gradient_descent(Network *nn, int layer, double *input_layer, double *output_layer_single, double *delta, double *prev_activated_layer) ;
//void grad_desc_regular(Network *nn, int layer, double *input_layer, double *output_layer_single, double *delta, double *prev_activated_layer) ;
double activationfn(double x) ;
double activationfnDeriv(double x) ;
void calculateOut(double *output, int output_classes) ;
void test_dataset(Network *nn, double **matrix_input, double **output_layer, int train_data, int test_num) ;
void show_weights(Network *nn) ;


