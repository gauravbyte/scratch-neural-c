#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<string.h>
#include <string.h>

double moment = 0.9 ;
double learningrate = 0.1 ;
double Epsilon = 0.000000001 ;
int act_hid = 1;
double bias = 0.000001;
int actfnid ;
int test_test_num ; 

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
    double activationfn(double x) ;
    double activationfnDeriv(double x) ;
    void calculateOut(double *output, int output_classes) ;
    void test_dataset(Network *nn, double **matrix_input, double **output_layer, int train_data, int test_num) ;
    void show_weights(Network *nn) ;

int main(int argc , char **argv)
{
int *neuronlist;
double **matrix_input,**output_layer ;
int train_data ;
FILE *filetok ;
int epochs;
int *neuron_count ;
int total_data = 0 ;
int feature_count , countlayer ;
countlayer= atoi(argv[1]);
if(countlayer < 0){
printf("total layers must be grater than");
exit(0);
}


neuronlist = (int*)malloc(countlayer * sizeof(int));

char* toki;
toki = strtok(argv[2], ",");
for(int i =0; i < countlayer; i++){
neuronlist[i] = atoi(toki);
if(neuronlist[i] <= 0){
    printf("Hidden layer sizes should be positive");
    exit(0);
}
toki = strtok(NULL, ",");
}

actfnid = atoi(argv[3]);
epochs = atoi(argv[4]);
learningrate = atof(argv[5]);
int k = 0 ;
double err ;
srand(100) ;



filetok = fopen("cancer.txt" ,"r") ;

total_data =567;
matrix_input = (double **)malloc(sizeof(double *) * total_data) ;
output_layer = (double **)malloc(sizeof(double *) * total_data) ;
train_data = 400 ;
feature_count = 30 ;

filetok = fopen("cancer.txt" ,"r") ;
if(filetok != NULL)
{
    for(int i = 0 ; i < total_data ; i++)
    {
        matrix_input[i] = (double *)malloc(sizeof(double) * 30) ;
        output_layer[i] = (double *)malloc(sizeof(double) * 2) ;
        double t;
        // fscanf(filetok, "%lf ,%lf,%lf,%lf ,%lf\n", &(matrix_input[i][0]), &(matrix_input[i][1]), &red, &(matrix_input[i][2]), &temp) ;
        // printf("line num %d         ",i);
            for(int j=0; j < 30; j++)
        {
            fscanf(filetok,"%lf", &(matrix_input[i][j]));
            // printf("  %lf  ",(matrix_input[i][j]));

        }
        fscanf(filetok,"%lf\n",&t);
        // printf("  %lf \n",t);
        if(t == 0)
        {
            output_layer[i][0] = 1 ;
            output_layer[i][1] = 0 ;
        }    
        else
        {
            output_layer[i][1] = 1 ;
            output_layer[i][0] = 0 ;
        }    
            
    }    
}
fclose(filetok) ;
//normalyzing dataset

double *array_max = (double *)malloc(sizeof(double) * 30) ;
for(int i = 0 ; i < 3 ; i++)
    array_max[i] = 0.0 ;
for(int i = 0 ; i < total_data ; i++)
{
    for(int j = 0 ; j < 30 ; j++ )
    {
        if(array_max[j] < matrix_input[i][j])
            array_max[j] = matrix_input[i][j] ;
    }
}
for(int i = 0 ; i < total_data ; i++)
{
    for(int j = 0 ; j < 30 ; j++ )
    {
        matrix_input[i][j] = matrix_input[i][j] / (double)array_max[j] ;
        matrix_input[i][j] -= 1 ;
    }
}
    total_data = 567;

//initialization of network      
Network *nn = (Network *)malloc(sizeof(Network)) ;

Layer *arr_layer ;
arr_layer = (Layer *)malloc(sizeof(Layer) * countlayer) ;
nn->layer_count = countlayer ;
nn->feature_count = feature_count ;
nn->arr_layer = arr_layer ;

int prev_neuron_count = feature_count ;
for(int i = 1 ; i <= countlayer ; i++)
{
    int count_neuron ;
    count_neuron = neuronlist[i-1];
    arr_layer[i-1].neuron_count = count_neuron ;
    arr_layer[i-1].neuraon_list = (Neuron *)malloc(sizeof(Neuron) * count_neuron) ;

    for(int j = 0 ; j < count_neuron ; j++)
    {
        Neuron *curr_neuron = arr_layer[i-1].neuraon_list + j ;
        curr_neuron->size = prev_neuron_count + 1 ;
        curr_neuron->list_weight = (double *)malloc(sizeof(double) * curr_neuron->size) ;
        curr_neuron->prev_list_weight = (double *)malloc(sizeof(double) * curr_neuron->size) ;

        for(int k = 0 ; k < curr_neuron->size ; k++)
        {
            curr_neuron->list_weight[k] = (double)rand()/(double) RAND_MAX ;
            curr_neuron->prev_list_weight[k] = 0.0 ;
        }
    }
    prev_neuron_count = count_neuron ;
}

FILE *fptr = fopen("output.txt", "w");
if (fptr == NULL)
{
    printf("Could not open file");
    return 0;
}

do{
    k++ ;
        err = mean_squared_err(nn,matrix_input,output_layer,train_data) ;
    // err = cross_entropy_loss(nn,matrix_input,output_layer,train_data) ;
    
    if(k % 10==0){
    printf("\nepochs %d  ,loss value %lf ",k , err) ;
    fprintf(fptr,"\n%d   %lf ",k,err);
    } 
    // printf("\nepoch :%d  celoss: %lf ",k , err) ;

    for(int i = 0 ; i < train_data ; i++)
    {
        for(int j = nn->layer_count-1 ; j >= 0 ; j--)
        {
            double *prev_activated_layer = j>0 ? forward_layer(nn,matrix_input[i],j,1) : NULL ;
            double *delta = calculate_delta(nn,j+1,matrix_input[i],output_layer[i]) ;
            gradient_descent(nn,j,matrix_input[i],output_layer[i],delta,prev_activated_layer) ;    
        }
    }

}
while(k <epochs);
printf("\nEpochs : %d",k) ;

test_dataset(nn,matrix_input,output_layer,total_data,train_data+1) ;
fclose(fptr);
}

double mean_squared_err(Network *nn,double **matrix_input,double **output_layer,int train_data) 
{

    double err = 0.0 ;
    int class_num = nn->arr_layer[nn->layer_count-1].neuron_count ;
    
    for(int i = 0 ; i < train_data ; i++)
    {
        double *calc_out = forward_layer(nn,matrix_input[i],nn->layer_count,1) ;
        double e = 0.0 ;
        
        for(int j = 0 ; j < class_num ; j++ )
        {
            e += pow(output_layer[i][j] - calc_out[j],2) ;
        }

        err += 0.5*e ;
    }    

    return err/train_data ;

}
double cross_entropy_loss(Network *nn,double **matrix_input,double **output_layer,int train_data) 
{

    double err = 0.0 ;
    int class_num = nn->arr_layer[nn->layer_count-1].neuron_count ;
    
    for(int i = 0 ; i < train_data ; i++)
    {
        double *calc_out = forward_layer(nn,matrix_input[i],nn->layer_count,1) ;
        double e = 0.0 ;
        
        for(int j = 0 ; j < class_num ; j++ )
        {
        e += output_layer[i][j]*log(calc_out[j]);
        }

        err += e ;
    }    

    return -err/train_data ;

}
double * forward_layer(Network *nn,double *input_layer,int layer,int act)
{

    Layer *layer_num = nn->arr_layer + (layer-1) ;
    int final_layer ;
    
    if(layer == nn->layer_count)
        final_layer = 0 ;
    else
        final_layer = 1 ;

    if(act)
    {        
        int layer_size = layer_num->neuron_count + final_layer ;
        double *unactive_layer = forward_layer(nn,input_layer,layer,0) ;
        double *activated_layer = (double *) malloc(sizeof(double) * layer_size) ;

        for(int i = 0 ; i < layer_size ;i++){
            if(!final_layer)
                activated_layer[i] = activationfn(unactive_layer[i]) ;
            else
            { 
                act_hid = actfnid;
                actfnid =1;
                // activated_layer[i] = i==0 ? -1.0 : activationfn(unactive_layer[i-1]) ;
                //trying activationfn fix
                //changing final layer to sigmoid for working results
                activated_layer[i] = activationfn(unactive_layer[i-1]);
                actfnid = act_hid;
            }
        }
        return activated_layer ;
    }
    else
    {
        int layer_size = layer_num->neuron_count , neuron_size = layer_num->neuraon_list->size ;
        double *unactive_layer = (double *) malloc(sizeof(double)* layer_size ) ;
        double *prev_layer = layer <= 1 ? input_layer : forward_layer(nn,input_layer,layer-1,1) ;

        for(int i=0 ; i < layer_size ; i++)
        {
            unactive_layer[i] = 0.0 ;
            Neuron *curr_neuron = layer_num->neuraon_list + i ;
            for(int j = 0 ; j < neuron_size ; j++)
            {
                    unactive_layer[i] += curr_neuron->list_weight[j] * prev_layer[j] ;
            }
            unactive_layer[i] += bias;
        }
        return unactive_layer ;
    }
    
}
//delta calculation of function
double * calculate_delta(Network *nn, int layer, double *input_layer, double *output_layer_single)
{
    
    int layer_size = (nn->arr_layer + layer - 1)->neuron_count ; 
    double *delta = (double *) malloc(sizeof(double) * layer_size);
    double *unactive_layer = forward_layer(nn, input_layer, layer , 0) ;

    if(nn->layer_count == layer)
    {
        double *activated_layer = forward_layer(nn, input_layer, layer , 1) ;
        act_hid = actfnid;
        actfnid = 1;
        for(int i = 0 ; i < layer_size ; i++)
            delta[i] = (output_layer_single[i] - activated_layer[i]) * activationfnDeriv(unactive_layer[i]);
        actfnid = act_hid    ;

    } 
    else 
    {
        double *next_delta = calculate_delta(nn, layer+1, input_layer, output_layer_single) ;
        Layer *next_layer = nn->arr_layer + layer ;

        for(int i = 0 ; i < layer_size ; i++){
            delta[i]=0.0 ;

            for(int j = 0 ; j < next_layer->neuron_count ; j++)
                delta[i] += next_delta[j] * *((next_layer->neuraon_list + j)->list_weight + i + 1) ;

            delta[i] *= activationfnDeriv(unactive_layer[i]) ;
        }

    }

    return delta;

}
//running gradient escent in backprop function
void gradient_descent(Network *nn, int layer, double *input_layer, double *output_layer_single, double *delta, double *prev_activated_layer)
{
    
    Layer *layer_num = nn->arr_layer + layer ;
    int layer_size = layer_num->neuron_count ;

    for(int i = 0 ; i < layer_size ; i++)
    {
        Neuron *curr_neuron = layer_num->neuraon_list + i ;

        for(int j = 0 ; j < curr_neuron->size ; j++)
        {
            
            double mom_term = moment * (*(curr_neuron->list_weight + j) - *(curr_neuron->prev_list_weight + j)) ;
            if( *(curr_neuron->prev_list_weight + j) == 0)
                mom_term = 0 ;
            *(curr_neuron->prev_list_weight + j) = *(curr_neuron->list_weight + j) ;
            double grad_desc_step = (1 - moment) * (learningrate * delta[i] * (layer==0 ? input_layer[j] : prev_activated_layer[j])) ;
            *(curr_neuron->list_weight + j) = *(curr_neuron->list_weight + j) + mom_term + grad_desc_step ;

        }

    }

}

double activationfn(double x){
    
    if(actfnid == 1)
    {
        return 1.0/(double)(1.0+exp(-x)) ;
    }
    else if(actfnid == 2)
    {
        return tanh(x);
    }
    else if(actfnid == 3)
    {
        return fmax(x,0);
    }    
    else
    {
        return 0.0 ;
    }
    
}

double activationfnDeriv(double x){
    
    if(actfnid == 1)
    {
        //Derivative of Logistic
        return activationfn(x)*(1.0-activationfn(x)) ;
    }
    else if(actfnid == 2)
    {
        // Derivative of Tanh
        double sech = 1.0 / cosh(x) ;
        return sech*sech ;
        // return 1.0-pow(activationfn(x),2);
        //don't know why but above implementation but changed implementation works
    }
        else if(actfnid == 3 )
    {//relu derivative
        if(x>0)
        return 1;
        else if(x<=0)
        return 0;
        // else 
    }
    else
    {
        return 0.0 ;
    }
    
}

void calculateOut(double *output, int output_classes)
{
    double out_max = -100000000;

    for(int i = 0 ; i < output_classes ; i++)
    {
        if(output[i] > out_max)
            out_max = output[i] ;
    }
    for(int i = 0 ; i < output_classes ; i++)
    {
        if(output[i] == out_max)
            output[i] = 1 ;
        else
            output[i] = 0 ;
    }

}
//test dataset for given tests
void test_dataset(Network *nn, double **matrix_input, double **output_layer, int train_data, int test_num) 
{
    int correct_pred = 0 ;

    for(int i = test_num-1 ; i < train_data ; i++)
    {
        int output_classes = (nn->arr_layer + (nn->layer_count - 1))->neuron_count ;
        double *out_wo_act = forward_layer(nn,matrix_input[i],nn->layer_count ,1) ;
        // printf("data be %d, %lf   %lf expected %lf  %lf\n",i,out_wo_act[0],out_wo_act[1],output_layer[i][0],output_layer[i][1]);

        calculateOut(out_wo_act,output_classes) ;
        //printf("data no %d, %lf   %lf expected %lf  %lf\n",i,out_wo_act[0],out_wo_act[1],output_layer[i][0],output_layer[i][1]);

        for(int j = 0 ; j < output_classes ; j++)
        {
            if(output_layer[i][j] == 1 && out_wo_act[j] == 1)
                correct_pred++ ;
        }
    }

    printf("\n final test accuracy %lf percent.",((double)correct_pred/(train_data-test_num+1))*100 ) ;
    
}
//to show weights of each layer and each neuron
void show_weights(Network *nn)
{
    int num_layers = nn->layer_count ;
    for(int i = 0 ; i< num_layers ; i++)
    {
        Layer *layer_num = nn->arr_layer + i ;
        int neuron_count = layer_num->neuron_count ;
        printf("\nLayer %d : ",i+1) ;
        for(int j = 0 ; j < neuron_count ; j++ )
        {
            Neuron *curr_neuron = layer_num->neuraon_list + j ;
            int weights_num = curr_neuron->size ;
            printf("\n\tNeuron %d : ", j+1 ) ;
            for(int k = 0 ; k < weights_num ; k++)
            {
                printf("\t %lf" ,*(curr_neuron->list_weight + k) ) ;
            }
        }
    }
}