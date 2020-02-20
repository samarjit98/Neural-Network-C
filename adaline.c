#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct{
	double *weights; // layer x dim curr layer x dim prev layer
	double bias; // layer x dim curr layer

    double *delta_weights;
    double delta_bias;
} linear_t;
linear_t adaline;
int num_weights;
int num_epochs;
double learning_rate;
int batch_size;

// data for OR gate
int features[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
//int labels[4] = {0, 1, 1, 1};
int labels[4] = {0, 0, 0, 1}; // labels for AND gate

double gaussrand(){
	static double V1, V2, S;
	static int phase = 0;
	double X;
	if(phase == 0) {
		do {
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else X = V2 * sqrt(-2 * log(S) / S);
	phase = 1 - phase;
	return X;
}

double signum(double n){
    if(n>0.0)return 1.0;
    else return -1.0;
}

double forward(int batch_feature[][2], int batch_label[]){
    double error;

    error = 0.0;

    for(int i=0; i<num_weights; i++)adaline.delta_weights[i] = 0.0;
    adaline.delta_bias = 0.0;

    for(int i=0; i<batch_size; i++){
        double curr_hyp = 0.0;

        for(int j=0; j<num_weights; j++)curr_hyp += adaline.weights[j]*(double)batch_feature[i][j];
        curr_hyp += adaline.bias;

        //printf("%lf\n", curr_hyp);

        double curr_error = (signum(curr_hyp) - signum((double)batch_label[i]))*(signum(curr_hyp) - signum((double)batch_label[i])) / 2.0;
        error += curr_error;

        for(int j=0; j<num_weights; j++)adaline.delta_weights[j] += learning_rate * (signum(curr_hyp) - signum((double)batch_label[i])) * (double)batch_feature[i][j];
        adaline.delta_bias += learning_rate * (signum(curr_hyp) - signum((double)batch_label[i]));
    }

    return error;
}

void backward(){
    for(int i=0; i<num_weights; i++)adaline.weights[i] += adaline.delta_weights[i];
    adaline.bias += adaline.delta_bias;
    
    printf("New weights: ");
    for(int i=0; i<num_weights; i++)printf("%lf ", adaline.weights[i]);
    printf("New Bias: %lf ", adaline.bias);
    printf("\n");
}

int main(int argc, char* argv[]){
    num_epochs = atoi(argv[1]);
    batch_size = atoi(argv[2]);
    learning_rate = atof(argv[3]);
    num_weights = atoi(argv[4]);
    
    adaline.weights = (double*)malloc(num_weights*sizeof(double));
    adaline.delta_weights = (double*)malloc(num_weights*sizeof(double));

    for(int i=0; i<num_weights; i++)adaline.weights[i] = 0.1;
    adaline.bias = 0.1;

    for(int i=0; i<num_epochs; i++){
        int num_batches = 4 / batch_size ; // 4 is dataset size

        for(int j=0; j<num_batches; j++){
            int batch_feature[batch_size][num_weights];
            int batch_label[batch_size];

            for(int k=0; k<batch_size; k++){
                batch_label[k] = labels[batch_size*j + k];
                for(int l=0; l<num_weights; l++)
                    batch_feature[k][l] = features[batch_size*j + k][l];
            }

            double error = forward(batch_feature, batch_label);
            backward();
            printf("Epoch [%d/%d], Batch [%d/%d], Error: %lf \n", i+1, num_epochs, j+1, num_batches, error);
        }
        printf("\n");
    }

    return 0;
}