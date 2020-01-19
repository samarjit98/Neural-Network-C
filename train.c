#include "mnist.h"

typedef struct{
	double ***weights;
	double **bias;
} network_t;

int main(void)
{
    load_mnist();

    // print pixels of first data in test dataset
    int i;
    for (i=0; i<784; i++) {
        printf("%1.1f ", test_image[0][i]);
        if ((i+1) % 28 == 0) putchar('\n');
    }

    // print first label in test dataset
    printf("label: %d\n", test_label[0]);

    int num_layers;
    scanf("%d", &num_layers);

    int *neurons;
    neurons = (int*)malloc(num_layers*sizeof(int));

    for(int i=0; i<num_layers; i++){
    	scanf("%d", &neurons[i]);
    }

    network_t network;
    network.weights = (double***)malloc(num_layers*sizeof(double**));
    network.bias = (double**)malloc(num_layers*sizeof(double*));

    for(int i=1; i<num_layers; i++){
    	network.weights[i] = (double**)malloc(neurons[i]*sizeof(double*));
    	network.bias[i] = (double*)malloc(neurons[i]*sizeof(double));

    	for(int j=0; j<neurons[i]; j++)network.weights[i][j] = (double*)malloc(neurons[i-1]*sizeof(double));
    }

    return 0;
}