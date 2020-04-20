#include "mnist.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

typedef struct{
    double ***weights; // layer x dim curr layer x dim prev layer
    double **bias; // layer x dim curr layer

 //   double ***x;
    double ***z; // batch x layer x dim curr layer
    double ***a; // batch x layer x dim curr layer
    double ****delta; // batch x layer x dim curr layer x dim prev layer
    double ***delta_b; 
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

double forward(double batch_image[][SIZE], int batch_label[], double* accuracy){
    double error = 0.0;
    double total = 0.0;
    for(int i=0; i<batch_size; i++){

        double* image;
        image = (double*)malloc(neurons[0]*sizeof(double));
        for(int ii=0; ii<SIZE; ii++)image[ii] = batch_image[i][ii];

        for(int ii=0; ii<neurons[0]; ii++)
            network.z[i][0][ii] =image[ii];

        for(int j=1; j<num_layers; j++){

            for(int ii=0; ii<neurons[j]; ii++){
                network.z[i][j][ii] = 0.0;
                //network.x[i][j][ii] = 0.0;
                for(int jj=0; jj<neurons[j-1]; jj++){
                    network.z[i][j][ii] += image[jj] * network.weights[j][ii][jj];
                }
                network.z[i][j][ii] += network.bias[j][ii];
                network.z[i][j][ii] = sigmoid(network.z[i][j][ii]);
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

        int predicted = 0;
        double value = 0.0;
        for(int ii=0; ii<neurons[num_layers - 1]; ii++){
            if(image[ii] > value){
                value = image[ii]; predicted = ii;
            }
        }
        if(predicted == batch_label[i])total += 1.0;

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

    *accuracy = (total / (double)batch_size)*100;
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
                    network.delta[i][j+1][l][k] = learning_rate*network.a[i][j+1][l]*network.z[i][j][k];
                }
            }
        }
    }

    for(int i=0; i<batch_size; i++){
        for(int j=1; j<num_layers; j++){
            for(int k=0; k<neurons[j]; k++){
                network.delta_b[i][j][k] = learning_rate*network.a[i][j][k];
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

    for(int j=1; j<num_layers; j++){
        for(int k=0; k<neurons[j]; k++){
            double net_delta = 0;
            for(int i=0; i<batch_size; i++)
                net_delta += network.delta_b[i][j][k];
            network.bias[j][k] += net_delta;
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
    //network.x = (double***)malloc(batch_size*sizeof(double**));
    network.a = (double***)malloc(batch_size*sizeof(double**));

    for(int i=0; i<batch_size; i++){
        network.z[i] = (double**)malloc(num_layers*sizeof(double*));
        //network.x[i] = (double**)malloc(num_layers*sizeof(double*));
        network.a[i] = (double**)malloc(num_layers*sizeof(double*));
        for(int j=0; j<num_layers; j++){
            network.z[i][j] = (double*)malloc(neurons[j]*sizeof(double));
            //network.x[i][j] = (double*)malloc(neurons[j]*sizeof(double));
            network.a[i][j] = (double*)malloc(neurons[j]*sizeof(double));
        }
    }

    network.delta = (double****)malloc(batch_size*sizeof(double***));
    network.delta_b = (double***)malloc(batch_size*sizeof(double**));
    for(int i=0; i<batch_size; i++){
        network.delta[i] = (double***)malloc(num_layers*sizeof(double**));
        network.delta_b[i] = (double**)malloc(num_layers*sizeof(double*));
        for(int j=1; j<num_layers; j++){
            network.delta[i][j] = (double**)malloc(neurons[j]*sizeof(double*));
            network.delta_b[i][j] = (double*)malloc(neurons[j]*sizeof(double));

            for(int k=0; k<neurons[j]; k++)
                network.delta[i][j][k] = (double*)malloc(neurons[j-1]*sizeof(double));
        }
    }

    for(int i=0; i<num_epochs; i++){
        int num_batches = NUM_TRAIN / batch_size ;
        double cumulative = 0.0;

        for(int j=0; j<num_batches; j++){
            double batch_image[batch_size][SIZE];
            int batch_label[batch_size];

            for(int k=0; k<batch_size; k++){
                batch_label[k] = train_label[batch_size*j + k];
                for(int l=0; l<SIZE; l++)
                    batch_image[k][l] = train_image[batch_size*j + k][l];
            }

            clock_t t;
            double accuracy;
            t = clock();
            double error = forward(batch_image, batch_label, &accuracy);
            t = clock() - t;
            backward();
            cumulative += accuracy;
            
            printf("Epoch [%d/%d], Batch [%d/%d], Train Error: %lf, Train Accuracy: %lf perc , Cumulative: %lf perc, Time: %lf \n", i+1, num_epochs, j+1, num_batches, error, accuracy, cumulative/(j+1), ((double)t)/CLOCKS_PER_SEC);
        }
        printf("\n");
        /*
        FILE* weights;
        weights = fopen("weights.dat", "wb");
        fwrite(&network, sizeof(network), 1, weights);
        fclose(weights);
        */
        int num_batches_test = NUM_TEST / batch_size ;
        cumulative = 0.0;

        for(int j=0; j<num_batches_test; j++){
            double batch_image[batch_size][SIZE];
            int batch_label[batch_size];

            for(int k=0; k<batch_size; k++){
                batch_label[k] = test_label[batch_size*j + k];
                for(int l=0; l<SIZE; l++)
                    batch_image[k][l] = test_image[batch_size*j + k][l];
            }
            
            clock_t t;
            double accuracy;
            t = clock();
            double error = forward(batch_image, batch_label, &accuracy);
            t = clock() - t;
            cumulative += accuracy;

            printf("Epoch [%d/%d], Batch [%d/%d], Test Error: %lf, Test Accuracy: %lf perc , Cumulative: %lf perc, Time: %lf \n", i+1, num_epochs, j+1, num_batches_test, error, accuracy, cumulative/(j+1), ((double)t)/CLOCKS_PER_SEC);
        }
        printf("\n");
    }

    return 0;
}

/*
Batch-size 64 LR 0.01

7 layer network 784 500 250 125 75 50 10

Epoch [10/10], Batch [1/156], Test Error: 5.019670, Test Accuracy: 95.312500 perc , Cumulative: 95.312500 perc 
Epoch [10/10], Batch [2/156], Test Error: 5.770697, Test Accuracy: 95.312500 perc , Cumulative: 95.312500 perc 
Epoch [10/10], Batch [3/156], Test Error: 6.241203, Test Accuracy: 95.312500 perc , Cumulative: 95.312500 perc 
Epoch [10/10], Batch [4/156], Test Error: 6.747266, Test Accuracy: 93.750000 perc , Cumulative: 94.921875 perc 
Epoch [10/10], Batch [5/156], Test Error: 7.591436, Test Accuracy: 93.750000 perc , Cumulative: 94.687500 perc 
Epoch [10/10], Batch [6/156], Test Error: 8.317874, Test Accuracy: 89.062500 perc , Cumulative: 93.750000 perc 
Epoch [10/10], Batch [7/156], Test Error: 9.354410, Test Accuracy: 90.625000 perc , Cumulative: 93.303571 perc 
Epoch [10/10], Batch [8/156], Test Error: 11.507517, Test Accuracy: 87.500000 perc , Cumulative: 92.578125 perc 
Epoch [10/10], Batch [9/156], Test Error: 11.112788, Test Accuracy: 89.062500 perc , Cumulative: 92.187500 perc 
Epoch [10/10], Batch [10/156], Test Error: 8.154767, Test Accuracy: 93.750000 perc , Cumulative: 92.343750 perc 
Epoch [10/10], Batch [11/156], Test Error: 6.978360, Test Accuracy: 92.187500 perc , Cumulative: 92.329545 perc 
Epoch [10/10], Batch [12/156], Test Error: 7.746368, Test Accuracy: 92.187500 perc , Cumulative: 92.317708 perc 
Epoch [10/10], Batch [13/156], Test Error: 3.820399, Test Accuracy: 100.000000 perc , Cumulative: 92.908654 perc 
Epoch [10/10], Batch [14/156], Test Error: 8.815970, Test Accuracy: 90.625000 perc , Cumulative: 92.745536 perc 
Epoch [10/10], Batch [15/156], Test Error: 12.195782, Test Accuracy: 90.625000 perc , Cumulative: 92.604167 perc 
Epoch [10/10], Batch [16/156], Test Error: 9.750040, Test Accuracy: 92.187500 perc , Cumulative: 92.578125 perc 
Epoch [10/10], Batch [17/156], Test Error: 10.131197, Test Accuracy: 92.187500 perc , Cumulative: 92.555147 perc 
Epoch [10/10], Batch [18/156], Test Error: 8.973679, Test Accuracy: 92.187500 perc , Cumulative: 92.534722 perc 
Epoch [10/10], Batch [19/156], Test Error: 10.562576, Test Accuracy: 90.625000 perc , Cumulative: 92.434211 perc 
Epoch [10/10], Batch [20/156], Test Error: 13.437403, Test Accuracy: 85.937500 perc , Cumulative: 92.109375 perc 
Epoch [10/10], Batch [21/156], Test Error: 11.559004, Test Accuracy: 89.062500 perc , Cumulative: 91.964286 perc 
Epoch [10/10], Batch [22/156], Test Error: 5.309190, Test Accuracy: 96.875000 perc , Cumulative: 92.187500 perc 
Epoch [10/10], Batch [23/156], Test Error: 8.969991, Test Accuracy: 95.312500 perc , Cumulative: 92.323370 perc 
Epoch [10/10], Batch [24/156], Test Error: 8.312573, Test Accuracy: 92.187500 perc , Cumulative: 92.317708 perc 
Epoch [10/10], Batch [25/156], Test Error: 9.435968, Test Accuracy: 90.625000 perc , Cumulative: 92.250000 perc 
Epoch [10/10], Batch [26/156], Test Error: 7.480204, Test Accuracy: 95.312500 perc , Cumulative: 92.367788 perc 
Epoch [10/10], Batch [27/156], Test Error: 10.756938, Test Accuracy: 89.062500 perc , Cumulative: 92.245370 perc 
Epoch [10/10], Batch [28/156], Test Error: 10.233415, Test Accuracy: 87.500000 perc , Cumulative: 92.075893 perc 
Epoch [10/10], Batch [29/156], Test Error: 7.994601, Test Accuracy: 93.750000 perc , Cumulative: 92.133621 perc 
Epoch [10/10], Batch [30/156], Test Error: 9.754156, Test Accuracy: 90.625000 perc , Cumulative: 92.083333 perc 
Epoch [10/10], Batch [31/156], Test Error: 9.586854, Test Accuracy: 93.750000 perc , Cumulative: 92.137097 perc 
Epoch [10/10], Batch [32/156], Test Error: 11.540057, Test Accuracy: 90.625000 perc , Cumulative: 92.089844 perc 
Epoch [10/10], Batch [33/156], Test Error: 10.612435, Test Accuracy: 90.625000 perc , Cumulative: 92.045455 perc 
Epoch [10/10], Batch [34/156], Test Error: 9.949079, Test Accuracy: 90.625000 perc , Cumulative: 92.003676 perc 
Epoch [10/10], Batch [35/156], Test Error: 9.474376, Test Accuracy: 92.187500 perc , Cumulative: 92.008929 perc 
Epoch [10/10], Batch [36/156], Test Error: 7.814190, Test Accuracy: 93.750000 perc , Cumulative: 92.057292 perc 
Epoch [10/10], Batch [37/156], Test Error: 7.074622, Test Accuracy: 95.312500 perc , Cumulative: 92.145270 perc 
Epoch [10/10], Batch [38/156], Test Error: 11.941323, Test Accuracy: 90.625000 perc , Cumulative: 92.105263 perc 
Epoch [10/10], Batch [39/156], Test Error: 6.109006, Test Accuracy: 95.312500 perc , Cumulative: 92.187500 perc 
Epoch [10/10], Batch [40/156], Test Error: 5.924294, Test Accuracy: 96.875000 perc , Cumulative: 92.304688 perc 
Epoch [10/10], Batch [41/156], Test Error: 9.063725, Test Accuracy: 90.625000 perc , Cumulative: 92.263720 perc 
Epoch [10/10], Batch [42/156], Test Error: 8.526889, Test Accuracy: 92.187500 perc , Cumulative: 92.261905 perc 
Epoch [10/10], Batch [43/156], Test Error: 8.517237, Test Accuracy: 92.187500 perc , Cumulative: 92.260174 perc 
Epoch [10/10], Batch [44/156], Test Error: 8.678623, Test Accuracy: 90.625000 perc , Cumulative: 92.223011 perc 
Epoch [10/10], Batch [45/156], Test Error: 5.742493, Test Accuracy: 98.437500 perc , Cumulative: 92.361111 perc 
Epoch [10/10], Batch [46/156], Test Error: 9.011892, Test Accuracy: 92.187500 perc , Cumulative: 92.357337 perc 
Epoch [10/10], Batch [47/156], Test Error: 11.662004, Test Accuracy: 89.062500 perc , Cumulative: 92.287234 perc 
Epoch [10/10], Batch [48/156], Test Error: 5.623543, Test Accuracy: 95.312500 perc , Cumulative: 92.350260 perc 
Epoch [10/10], Batch [49/156], Test Error: 9.831374, Test Accuracy: 92.187500 perc , Cumulative: 92.346939 perc 
Epoch [10/10], Batch [50/156], Test Error: 9.298403, Test Accuracy: 95.312500 perc , Cumulative: 92.406250 perc 
Epoch [10/10], Batch [51/156], Test Error: 5.858531, Test Accuracy: 93.750000 perc , Cumulative: 92.432598 perc 
Epoch [10/10], Batch [52/156], Test Error: 5.555126, Test Accuracy: 96.875000 perc , Cumulative: 92.518029 perc 
Epoch [10/10], Batch [53/156], Test Error: 8.160867, Test Accuracy: 90.625000 perc , Cumulative: 92.482311 perc 
Epoch [10/10], Batch [54/156], Test Error: 7.061478, Test Accuracy: 93.750000 perc , Cumulative: 92.505787 perc 
Epoch [10/10], Batch [55/156], Test Error: 9.473103, Test Accuracy: 90.625000 perc , Cumulative: 92.471591 perc 
Epoch [10/10], Batch [56/156], Test Error: 11.920416, Test Accuracy: 89.062500 perc , Cumulative: 92.410714 perc 
Epoch [10/10], Batch [57/156], Test Error: 7.644555, Test Accuracy: 95.312500 perc , Cumulative: 92.461623 perc 
Epoch [10/10], Batch [58/156], Test Error: 5.127814, Test Accuracy: 96.875000 perc , Cumulative: 92.537716 perc 
Epoch [10/10], Batch [59/156], Test Error: 13.094241, Test Accuracy: 87.500000 perc , Cumulative: 92.452331 perc 
Epoch [10/10], Batch [60/156], Test Error: 17.133098, Test Accuracy: 81.250000 perc , Cumulative: 92.265625 perc 
Epoch [10/10], Batch [61/156], Test Error: 9.798278, Test Accuracy: 92.187500 perc , Cumulative: 92.264344 perc 
Epoch [10/10], Batch [62/156], Test Error: 10.260878, Test Accuracy: 90.625000 perc , Cumulative: 92.237903 perc 
Epoch [10/10], Batch [63/156], Test Error: 11.740967, Test Accuracy: 89.062500 perc , Cumulative: 92.187500 perc 
Epoch [10/10], Batch [64/156], Test Error: 9.998085, Test Accuracy: 89.062500 perc , Cumulative: 92.138672 perc 
Epoch [10/10], Batch [65/156], Test Error: 6.741398, Test Accuracy: 95.312500 perc , Cumulative: 92.187500 perc 
Epoch [10/10], Batch [66/156], Test Error: 12.208809, Test Accuracy: 89.062500 perc , Cumulative: 92.140152 perc 
Epoch [10/10], Batch [67/156], Test Error: 9.928623, Test Accuracy: 92.187500 perc , Cumulative: 92.140858 perc 
Epoch [10/10], Batch [68/156], Test Error: 10.832040, Test Accuracy: 87.500000 perc , Cumulative: 92.072610 perc 
Epoch [10/10], Batch [69/156], Test Error: 11.091610, Test Accuracy: 92.187500 perc , Cumulative: 92.074275 perc 
Epoch [10/10], Batch [70/156], Test Error: 6.716149, Test Accuracy: 93.750000 perc , Cumulative: 92.098214 perc 
Epoch [10/10], Batch [71/156], Test Error: 11.837137, Test Accuracy: 87.500000 perc , Cumulative: 92.033451 perc 
Epoch [10/10], Batch [72/156], Test Error: 7.674936, Test Accuracy: 93.750000 perc , Cumulative: 92.057292 perc 
Epoch [10/10], Batch [73/156], Test Error: 7.914762, Test Accuracy: 92.187500 perc , Cumulative: 92.059075 perc 
Epoch [10/10], Batch [74/156], Test Error: 8.146068, Test Accuracy: 92.187500 perc , Cumulative: 92.060811 perc 
Epoch [10/10], Batch [75/156], Test Error: 6.917227, Test Accuracy: 93.750000 perc , Cumulative: 92.083333 perc 
Epoch [10/10], Batch [76/156], Test Error: 11.013827, Test Accuracy: 89.062500 perc , Cumulative: 92.043586 perc 
Epoch [10/10], Batch [77/156], Test Error: 11.609878, Test Accuracy: 87.500000 perc , Cumulative: 91.984578 perc 
Epoch [10/10], Batch [78/156], Test Error: 7.412606, Test Accuracy: 92.187500 perc , Cumulative: 91.987179 perc 
Epoch [10/10], Batch [79/156], Test Error: 5.064229, Test Accuracy: 96.875000 perc , Cumulative: 92.049051 perc 
Epoch [10/10], Batch [80/156], Test Error: 3.086062, Test Accuracy: 98.437500 perc , Cumulative: 92.128906 perc 
Epoch [10/10], Batch [81/156], Test Error: 6.600674, Test Accuracy: 93.750000 perc , Cumulative: 92.148920 perc 
Epoch [10/10], Batch [82/156], Test Error: 4.601435, Test Accuracy: 96.875000 perc , Cumulative: 92.206555 perc 
Epoch [10/10], Batch [83/156], Test Error: 1.861032, Test Accuracy: 100.000000 perc , Cumulative: 92.300452 perc 
Epoch [10/10], Batch [84/156], Test Error: 2.255327, Test Accuracy: 98.437500 perc , Cumulative: 92.373512 perc 
Epoch [10/10], Batch [85/156], Test Error: 3.080971, Test Accuracy: 98.437500 perc , Cumulative: 92.444853 perc 
Epoch [10/10], Batch [86/156], Test Error: 3.310253, Test Accuracy: 98.437500 perc , Cumulative: 92.514535 perc 
Epoch [10/10], Batch [87/156], Test Error: 2.095051, Test Accuracy: 100.000000 perc , Cumulative: 92.600575 perc 
Epoch [10/10], Batch [88/156], Test Error: 9.596660, Test Accuracy: 90.625000 perc , Cumulative: 92.578125 perc 
Epoch [10/10], Batch [89/156], Test Error: 6.792415, Test Accuracy: 95.312500 perc , Cumulative: 92.608848 perc 
Epoch [10/10], Batch [90/156], Test Error: 6.496079, Test Accuracy: 93.750000 perc , Cumulative: 92.621528 perc 
Epoch [10/10], Batch [91/156], Test Error: 3.025899, Test Accuracy: 98.437500 perc , Cumulative: 92.685440 perc 
Epoch [10/10], Batch [92/156], Test Error: 6.429429, Test Accuracy: 95.312500 perc , Cumulative: 92.713995 perc 
Epoch [10/10], Batch [93/156], Test Error: 9.743823, Test Accuracy: 92.187500 perc , Cumulative: 92.708333 perc 
Epoch [10/10], Batch [94/156], Test Error: 11.674178, Test Accuracy: 87.500000 perc , Cumulative: 92.652926 perc 
Epoch [10/10], Batch [95/156], Test Error: 15.361470, Test Accuracy: 82.812500 perc , Cumulative: 92.549342 perc 
Epoch [10/10], Batch [96/156], Test Error: 6.750232, Test Accuracy: 93.750000 perc , Cumulative: 92.561849 perc 
Epoch [10/10], Batch [97/156], Test Error: 6.021132, Test Accuracy: 95.312500 perc , Cumulative: 92.590206 perc 
Epoch [10/10], Batch [98/156], Test Error: 0.968826, Test Accuracy: 100.000000 perc , Cumulative: 92.665816 perc 
Epoch [10/10], Batch [99/156], Test Error: 2.494075, Test Accuracy: 98.437500 perc , Cumulative: 92.724116 perc 
Epoch [10/10], Batch [100/156], Test Error: 6.887166, Test Accuracy: 92.187500 perc , Cumulative: 92.718750 perc 
Epoch [10/10], Batch [101/156], Test Error: 3.462583, Test Accuracy: 98.437500 perc , Cumulative: 92.775371 perc 
Epoch [10/10], Batch [102/156], Test Error: 4.617533, Test Accuracy: 96.875000 perc , Cumulative: 92.815564 perc 
Epoch [10/10], Batch [103/156], Test Error: 13.505046, Test Accuracy: 87.500000 perc , Cumulative: 92.763956 perc 
Epoch [10/10], Batch [104/156], Test Error: 12.956449, Test Accuracy: 85.937500 perc , Cumulative: 92.698317 perc 
Epoch [10/10], Batch [105/156], Test Error: 3.419710, Test Accuracy: 96.875000 perc , Cumulative: 92.738095 perc 
Epoch [10/10], Batch [106/156], Test Error: 10.489379, Test Accuracy: 89.062500 perc , Cumulative: 92.703420 perc 
Epoch [10/10], Batch [107/156], Test Error: 5.019004, Test Accuracy: 95.312500 perc , Cumulative: 92.727804 perc 
Epoch [10/10], Batch [108/156], Test Error: 1.383852, Test Accuracy: 100.000000 perc , Cumulative: 92.795139 perc 
Epoch [10/10], Batch [109/156], Test Error: 3.472327, Test Accuracy: 98.437500 perc , Cumulative: 92.846904 perc 
Epoch [10/10], Batch [110/156], Test Error: 2.745827, Test Accuracy: 98.437500 perc , Cumulative: 92.897727 perc 
Epoch [10/10], Batch [111/156], Test Error: 2.714270, Test Accuracy: 100.000000 perc , Cumulative: 92.961712 perc 
Epoch [10/10], Batch [112/156], Test Error: 3.709630, Test Accuracy: 96.875000 perc , Cumulative: 92.996652 perc 
Epoch [10/10], Batch [113/156], Test Error: 2.839310, Test Accuracy: 100.000000 perc , Cumulative: 93.058628 perc 
Epoch [10/10], Batch [114/156], Test Error: 4.420412, Test Accuracy: 98.437500 perc , Cumulative: 93.105811 perc 
Epoch [10/10], Batch [115/156], Test Error: 2.211114, Test Accuracy: 98.437500 perc , Cumulative: 93.152174 perc 
Epoch [10/10], Batch [116/156], Test Error: 1.692006, Test Accuracy: 100.000000 perc , Cumulative: 93.211207 perc 
Epoch [10/10], Batch [117/156], Test Error: 7.284685, Test Accuracy: 93.750000 perc , Cumulative: 93.215812 perc 
Epoch [10/10], Batch [118/156], Test Error: 5.959261, Test Accuracy: 96.875000 perc , Cumulative: 93.246822 perc 
Epoch [10/10], Batch [119/156], Test Error: 5.376054, Test Accuracy: 96.875000 perc , Cumulative: 93.277311 perc 
Epoch [10/10], Batch [120/156], Test Error: 3.514264, Test Accuracy: 98.437500 perc , Cumulative: 93.320312 perc 
Epoch [10/10], Batch [121/156], Test Error: 3.995622, Test Accuracy: 96.875000 perc , Cumulative: 93.349690 perc 
Epoch [10/10], Batch [122/156], Test Error: 3.896337, Test Accuracy: 96.875000 perc , Cumulative: 93.378586 perc 
Epoch [10/10], Batch [123/156], Test Error: 9.992904, Test Accuracy: 89.062500 perc , Cumulative: 93.343496 perc 
Epoch [10/10], Batch [124/156], Test Error: 5.635287, Test Accuracy: 95.312500 perc , Cumulative: 93.359375 perc 
Epoch [10/10], Batch [125/156], Test Error: 3.001865, Test Accuracy: 98.437500 perc , Cumulative: 93.400000 perc 
Epoch [10/10], Batch [126/156], Test Error: 3.348481, Test Accuracy: 98.437500 perc , Cumulative: 93.439980 perc 
Epoch [10/10], Batch [127/156], Test Error: 6.303287, Test Accuracy: 96.875000 perc , Cumulative: 93.467028 perc 
Epoch [10/10], Batch [128/156], Test Error: 1.854492, Test Accuracy: 100.000000 perc , Cumulative: 93.518066 perc 
Epoch [10/10], Batch [129/156], Test Error: 5.138925, Test Accuracy: 96.875000 perc , Cumulative: 93.544089 perc 
Epoch [10/10], Batch [130/156], Test Error: 7.915369, Test Accuracy: 93.750000 perc , Cumulative: 93.545673 perc 
Epoch [10/10], Batch [131/156], Test Error: 6.739882, Test Accuracy: 93.750000 perc , Cumulative: 93.547233 perc 
Epoch [10/10], Batch [132/156], Test Error: 6.683732, Test Accuracy: 93.750000 perc , Cumulative: 93.548769 perc 
Epoch [10/10], Batch [133/156], Test Error: 3.213566, Test Accuracy: 98.437500 perc , Cumulative: 93.585526 perc 
Epoch [10/10], Batch [134/156], Test Error: 5.816496, Test Accuracy: 93.750000 perc , Cumulative: 93.586754 perc 
Epoch [10/10], Batch [135/156], Test Error: 4.435975, Test Accuracy: 96.875000 perc , Cumulative: 93.611111 perc 
Epoch [10/10], Batch [136/156], Test Error: 2.351221, Test Accuracy: 98.437500 perc , Cumulative: 93.646599 perc 
Epoch [10/10], Batch [137/156], Test Error: 0.848048, Test Accuracy: 100.000000 perc , Cumulative: 93.692974 perc 
Epoch [10/10], Batch [138/156], Test Error: 0.963521, Test Accuracy: 100.000000 perc , Cumulative: 93.738678 perc 
Epoch [10/10], Batch [139/156], Test Error: 1.580783, Test Accuracy: 100.000000 perc , Cumulative: 93.783723 perc 
Epoch [10/10], Batch [140/156], Test Error: 2.087895, Test Accuracy: 98.437500 perc , Cumulative: 93.816964 perc 
Epoch [10/10], Batch [141/156], Test Error: 7.126039, Test Accuracy: 93.750000 perc , Cumulative: 93.816489 perc 
Epoch [10/10], Batch [142/156], Test Error: 5.628957, Test Accuracy: 96.875000 perc , Cumulative: 93.838028 perc 
Epoch [10/10], Batch [143/156], Test Error: 2.743127, Test Accuracy: 100.000000 perc , Cumulative: 93.881119 perc 
Epoch [10/10], Batch [144/156], Test Error: 4.230529, Test Accuracy: 98.437500 perc , Cumulative: 93.912760 perc 
Epoch [10/10], Batch [145/156], Test Error: 1.117424, Test Accuracy: 100.000000 perc , Cumulative: 93.954741 perc 
Epoch [10/10], Batch [146/156], Test Error: 5.027891, Test Accuracy: 95.312500 perc , Cumulative: 93.964041 perc 
Epoch [10/10], Batch [147/156], Test Error: 1.811838, Test Accuracy: 100.000000 perc , Cumulative: 94.005102 perc 
Epoch [10/10], Batch [148/156], Test Error: 5.967352, Test Accuracy: 93.750000 perc , Cumulative: 94.003378 perc 
Epoch [10/10], Batch [149/156], Test Error: 6.408072, Test Accuracy: 96.875000 perc , Cumulative: 94.022651 perc 
Epoch [10/10], Batch [150/156], Test Error: 2.825572, Test Accuracy: 98.437500 perc , Cumulative: 94.052083 perc 
Epoch [10/10], Batch [151/156], Test Error: 8.115784, Test Accuracy: 92.187500 perc , Cumulative: 94.039735 perc 
Epoch [10/10], Batch [152/156], Test Error: 10.581330, Test Accuracy: 89.062500 perc , Cumulative: 94.006990 perc 
Epoch [10/10], Batch [153/156], Test Error: 18.499489, Test Accuracy: 81.250000 perc , Cumulative: 93.923611 perc 
Epoch [10/10], Batch [154/156], Test Error: 6.881153, Test Accuracy: 95.312500 perc , Cumulative: 93.932630 perc 
Epoch [10/10], Batch [155/156], Test Error: 14.372037, Test Accuracy: 84.375000 perc , Cumulative: 93.870968 perc 
Epoch [10/10], Batch [156/156], Test Error: 10.469932, Test Accuracy: 89.062500 perc , Cumulative: 93.840144 perc 


*/