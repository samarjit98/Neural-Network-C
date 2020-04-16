#include "mnist.h"
#include <math.h>
#include <stdlib.h>

typedef struct{
    double ***weights; // layer x dim curr layer x dim prev layer
    double **bias; // layer x dim curr layer

    double ***x;
	double ***z; // batch x layer x dim curr layer
	double ***a; // batch x layer x dim curr layer
	double ****delta; // batch x layer x dim curr layer x dim prev layer
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

double sigmoid(double x){
     double exp_value;
     double return_value;

     exp_value = exp((double) -x);
     return_value = 1 / (1 + exp_value);

     return return_value;
}

double forward(double batch_image[][SIZE], int batch_label[]){
    double error = 0.0;

    for(int i=0; i<batch_size; i++){

        double* image;
        image = (double*)malloc(neurons[0]*sizeof(double));
        for(int ii=0; ii<SIZE; ii++)image[ii] = batch_image[i][ii];

        for(int ii=0; ii<neurons[0]; ii++)
            network.x[i][0][ii] =image[ii];

        for(int j=1; j<num_layers; j++){

            for(int ii=0; ii<neurons[j]; ii++){
                network.z[i][j][ii] = 0.0;
                network.x[i][j][ii] = 0.0;
                for(int jj=0; jj<neurons[j-1]; jj++){
                    network.x[i][j][ii] += image[jj] * network.weights[j][ii][jj];
                }
                network.z[i][j][ii] = sigmoid(network.x[i][j][ii]);
            }

            free(image);
            image = (double*)malloc(neurons[j]*sizeof(double));
            for(int ii=0; ii<neurons[j]; ii++)image[ii] = network.z[i][j][ii];
        }

        double* label;
        label = (double*)malloc(neurons[num_layers - 1]*sizeof(double));
        for(int ii=0; ii<neurons[num_layers - 1]; ii++){
            if(ii == batch_label[i])label[ii] = 1.0;
            else label[ii] = 0.0;
        }

        //for(int ii=0; ii<neurons[num_layers - 1]; ii++)image[ii] = sigmoid(image[ii]);

        /*
        printf("Label: ");
        for(int ii=0; ii<neurons[num_layers - 1]; ii++)printf("%lf ", label[ii]);
        printf("\n");
        printf("Predicted: ");
        for(int ii=0; ii<neurons[num_layers - 1]; ii++)printf("%lf ", image[ii]);
        printf("\n");
        */
        for(int ii=0; ii<neurons[num_layers - 1]; ii++)
            network.a[i][num_layers - 1][ii] = image[ii]*(1 - image[ii])*(label[ii] - image[ii]);

        double image_error = 0.0;
        for(int ii=0; ii<neurons[num_layers - 1]; ii++)image_error += (image[ii] - label[ii])*(image[ii] - label[ii]);

        image_error = sqrt(image_error);

        error += image_error;
    }

    return error;
}

void backward(){
    for(int i=num_layers-2; i>=1; i--){
        for(int j=0; j<batch_size; j++){
            for(int k=0; k<neurons[i]; k++){
                double error_term = 0.0;
                for(int ii=0; ii<neurons[i+1]; ii++)
                    error_term += network.weights[i+1][ii][k]*network.a[j][i+1][ii];

                network.a[j][i][k] = network.z[j][i][k]*(1 - network.z[j][i][k])*error_term;
            }
        }
    }

    for(int i=0; i<batch_size; i++){
        for(int j=0; j<num_layers-1; j++){
            for(int k=0; k<neurons[j]; k++){
                for(int l=0; l<neurons[j+1]; l++){
                    network.delta[i][j+1][l][k] = learning_rate*network.a[i][j+1][l]*network.x[i][j][k];
                }
            }
        }
    }

    for(int j=0; j<num_layers-1; j++){
        for(int k=0; k<neurons[j]; k++){
            for(int l=0; l<neurons[j+1]; l++){
                double net_delta = 0;
                for(int i=0; i<batch_size; i++)
                    net_delta += network.delta[i][j+1][l][k];
                //  printf("%lf\n", net_delta);
                network.weights[j+1][l][k] += net_delta;
            }
        }
    }
}

int main(int argc, char* argv[]){
    load_mnist();

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

    network.z = (double***)malloc(batch_size*sizeof(double**));
    network.x = (double***)malloc(batch_size*sizeof(double**));
    network.a = (double***)malloc(batch_size*sizeof(double**));

    for(int i=0; i<batch_size; i++){
        network.z[i] = (double**)malloc(num_layers*sizeof(double*));
        network.x[i] = (double**)malloc(num_layers*sizeof(double*));
        network.a[i] = (double**)malloc(num_layers*sizeof(double*));
        for(int j=0; j<num_layers; j++){
            network.z[i][j] = (double*)malloc(neurons[j]*sizeof(double));
            network.x[i][j] = (double*)malloc(neurons[j]*sizeof(double));
            network.a[i][j] = (double*)malloc(neurons[j]*sizeof(double));
        }
    }

    network.delta = (double****)malloc(batch_size*sizeof(double***));
    for(int i=0; i<batch_size; i++){
        network.delta[i] = (double***)malloc(num_layers*sizeof(double**));
        for(int j=1; j<num_layers; j++){
            network.delta[i][j] = (double**)malloc(neurons[j]*sizeof(double*));

            for(int k=0; k<neurons[j]; k++)
                network.delta[i][j][k] = (double*)malloc(neurons[j-1]*sizeof(double));
        }
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
            backward();
            printf("Epoch [%d/%d], Batch [%d/%d], Error: %lf \n", i+1, num_epochs, j+1, num_batches, error);
        }
    }

    return 0;
}