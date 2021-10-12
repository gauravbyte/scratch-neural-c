#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>


int main()
{
       filetok = fopen("cancer.txt" ,"r") ;

    int total_data =567;
    int matrix_input = (double **)malloc(sizeof(double *) * total_data) ;
    int output_layer = (double **)malloc(sizeof(double *) * total_data) ;
    int train_data = 400 ;
    int feature_count = 30 ;
   
    filetok = fopen("cancer.txt" ,"r") ;
    if(filetok != NULL)
    {
        for(int i = 0 ; i < total_data ; i++)
        {
            matrix_input[i] = (double *)malloc(sizeof(double) * 30) ;
            output_layer[i] = (double *)malloc(sizeof(double) * 2) ;
            double t;
            // fscanf(filetok, "%lf ,%lf,%lf,%lf ,%lf\n", &(matrix_input[i][0]), &(matrix_input[i][1]), &red, &(matrix_input[i][2]), &temp) ;
            printf("line num %d         ",i);
                for(int j=0; j < 30; j++)
            {
                fscanf(filetok,"%lf", &(matrix_input[i][j]));
                printf("  %lf  ",(matrix_input[i][j]));

            }
            fscanf(filetok,"%lf\n",&t);
            printf("  %lf \n",t);
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
}