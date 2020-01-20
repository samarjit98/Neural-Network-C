#include "mnist.h"
#include <math.h>
#include <stdlib.h>

typedef struct{
	double ***weights; // layer x dim curr layer x dim prev layer
	double **bias; // layer x dim curr layer

	double ***z; // batch x layer x dim curr layer
	double ***a; // batch x layer x dim curr layer
	double ***delta; // batch x layer x dim curr layer
} network_t;
network_t network;
int num_layers;
int *neurons;
int batch_size, num_epochs;
double learning_rate;

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

double forward(double batch_image[][SIZE], int batch_label[]){
    double error;

    return error;
}

void backward(){

}

int main(int argc, char* argv[]){
    load_mnist();

    // print pixels of first data in test dataset
    int i;
    for (i=0; i<784; i++) {
        printf("%1.1f ", test_image[0][i]);
        if ((i+1) % 28 == 0) putchar('\n');
    }

    // print first label in test dataset
    printf("label: %d\n", test_label[0]);

    batch_size = atoi(argv[2]);
    num_epochs = atoi(argv[1]);
    learning_rate = atof(argv[3]);
    // scanf("%d", &batch_size); scanf("%d", &learning_rate);

    printf("Num Layers: ");
    scanf("%d", &num_layers);

    neurons = (int*)malloc(num_layers*sizeof(int));

    for(int i=0; i<num_layers; i++){
    	printf("Neurons Layer %d: ", i+1);
    	scanf("%d", &neurons[i]);
    }

    network.weights = (double***)malloc(num_layers*sizeof(double**));
    network.bias = (double**)malloc(num_layers*sizeof(double*));

    for(int i=1; i<num_layers; i++){
    	network.weights[i] = (double**)malloc(neurons[i]*sizeof(double*));
    	network.bias[i] = (double*)malloc(neurons[i]*sizeof(double));

    	for(int j=0; j<neurons[i]; j++)
    		network.weights[i][j] = (double*)malloc(neurons[i-1]*sizeof(double));

    	for(int j=0; j<neurons[i]; j++)
    		network.bias[i][j] = gaussrand();

    	for(int j=0; j<neurons[i]; j++)
    		for(int k=0; k<neurons[i-1]; k++)
    			network.weights[i][j][k] = gaussrand();
    }

    for(int i=0; i<num_epochs; i++){
        int num_batches = NUM_TRAIN / batch_size ;

        for(int j=0; j<num_batches; j++){
            double batch_image[batch_size][SIZE];
            int batch_label[batch_size];

            for(int k=0; k<batch_size; k++){
                batch_label[k] = train_label[batch_size*j + k];
                for(int l=0; l<SIZE; l++)
                    batch_image[k][l] = train_image[batch_size*j + k][l];
            }

            double error = forward(batch_image, batch_label);
            printf("Epoch [%d/%d], Batch [%d/%d], Error: %lf \n", i+1, num_epochs, j+1, num_batches, error);
        }
    }

    return 0;
}