#include "mnist.h"
#include <math.h>
#include <stdlib.h>
#include <pthread.h>

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

double** batch_image;
int* batch_label; 
double accuracy;
double error; pthread_mutex_t error_lock;
double total; pthread_mutex_t total_lock;

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

void* process_batch(void* argt){
    int i = *((int*)argt);
    //printf("Inside: %d\n", i);
    double* image;
    image = (double*)malloc(neurons[0]*sizeof(double));
    for(int ii=0; ii<SIZE; ii++)image[ii] = batch_image[i][ii];

    for(int ii=0; ii<neurons[0]; ii++)
        network.z[i][0][ii] = image[ii];

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
    if(predicted == batch_label[i]){
        pthread_mutex_lock(&total_lock);
        total += 1.0; // shared variable
        pthread_mutex_unlock(&total_lock);
    }

    for(int ii=0; ii<neurons[num_layers - 1]; ii++)
        network.a[i][num_layers - 1][ii] = image[ii]*(1 - image[ii])*(label[ii] - image[ii]);

    double image_error = 0.0;
    for(int ii=0; ii<neurons[num_layers - 1]; ii++)image_error += (image[ii] - label[ii])*(image[ii] - label[ii]);

    image_error = sqrt(image_error);
    
    pthread_mutex_lock(&error_lock);
    error += image_error; // shared variable
    pthread_mutex_unlock(&error_lock);

    free(image);
    free(label);

    //printf("Exited: %d\n", i);
}

void forward(){
    error = 0.0;
    total = 0.0;
    pthread_t* tid;
    tid = (pthread_t*)malloc(batch_size*sizeof(pthread_t));
    int* batch_id = (int*)malloc(batch_size*sizeof(int));
    for(int i=0; i<batch_size; i++){
        batch_id[i] = i;
        //printf("Starting: %d\n", i);
        pthread_create(&tid[i], NULL, process_batch, (void*)&batch_id[i]);
    }
    for(int i=0; i<batch_size; i++){
        pthread_join(tid[i], NULL);
    }
    free(tid);
    accuracy = (total / (double)batch_size)*100;
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
    pthread_mutex_init(&error_lock, NULL);
    pthread_mutex_init(&total_lock, NULL);

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

    batch_image = (double**)malloc(batch_size*sizeof(double*));
    for(int i=0; i<batch_size; i++)
        batch_image[i] = (double*)malloc(SIZE*sizeof(double));
    batch_label = (int*)malloc(batch_size*sizeof(int));

    for(int i=0; i<num_epochs; i++){
        int num_batches = NUM_TRAIN / batch_size ;
        double cumulative = 0.0;

        for(int j=0; j<num_batches; j++){
            for(int k=0; k<batch_size; k++){
                batch_label[k] = train_label[batch_size*j + k];
                for(int l=0; l<SIZE; l++)
                    batch_image[k][l] = train_image[batch_size*j + k][l];
            }
            clock_t t;
            t = clock();
            forward();
            t = clock() - t;
            cumulative += accuracy;
            backward();
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
            for(int k=0; k<batch_size; k++){
                batch_label[k] = test_label[batch_size*j + k];
                for(int l=0; l<SIZE; l++)
                    batch_image[k][l] = test_image[batch_size*j + k][l];
            }
            
            forward();
            cumulative += accuracy;

            printf("Epoch [%d/%d], Batch [%d/%d], Test Error: %lf, Test Accuracy: %lf perc , Cumulative: %lf perc \n", i+1, num_epochs, j+1, num_batches_test, error, accuracy, cumulative/(j+1));
        }
        printf("\n");
    }
    pthread_mutex_destroy(&error_lock); 
    pthread_mutex_destroy(&total_lock); 

    return 0;
}

