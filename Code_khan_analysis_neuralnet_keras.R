---
  #title: "Classification OF DATA_KHAN with the NEURALNET method"
  # author: "AREZKI Rafik"
 
set.seed(123) 
####-2.1- Data uploading:

# The data source
w<- getwd()

train_data_path<- paste(w,"data_Khan_train.RData",sep="/")
test_data_path<- paste(w,"data_Khan_test.RData",sep="/")



## Uploading training data :
load(train_data_path)

# Input data training
data_train<-data_Khan$X

# Output data training
data_train_labels<-data_Khan$Y

## Uploading test data :


## Test data :
load(test_data_path)
#Input data test

data_test<-data_Khan_test$X

# Output data test

data_test_labels<-data_Khan_test$Y

####-2.2- Data display:

data.frame(data_test_labels[1:5],data_train[1:5,1:6])


####-2.3- Descriptive analysis on learning data:
# explanatory variables
summary(data_train[,1:5])
# Variable to explain
data.frame(table(data_train_labels))

###-3- Part 01: Application of the * neuralnet * method :
####-3.1- Construction and recoding of the label matrix:

#Loading the library keras
library(keras)

C_data_train_labels <-to_categorical(data_train_labels, 2)

# Display of labels matrix
data.frame(Y_0=C_data_train_labels[20:30,1],Y_1=C_data_train_labels[20:30,2])

####-3.2- Writing the regression formula:
formula<-paste(paste(paste("Y_",0:1,sep = ''),collapse = '+'),'~',paste(paste('X',1:ncol(data_train),sep=''), collapse='+'), sep='')

#### - 3.3- Application of the * neuralnet * command :


data<-data.frame(C_data_train_labels,data_train)
colnames(data)<-c(paste("Y_",0:1,sep = ''),paste('X',1:ncol(data_train),sep=''))

#Loading the library neuralnet
library(neuralnet)
nn <- neuralnet(formule,data=data,hidden=c(150),lifesign = "minimal",threshold=0.01, linear.output=FALSE)

####-3.5- Prediction:

predict_test <- compute(nn,data_test)
pred_<-as.matrix(predict_test$net.result)
pr.nn<- max.col(pred_)-1

#####-3.5.1- The vector of predictions :
cat('Le vecteur prédit: ',pr.nn)

####-3.6- The confusion matrix:
library(caret)
confusionMatrix(as.factor(data_test_labels),as.factor(pr.nn))

####-3.7- Graphical visualization of predicted data:
plot(predict_test$net.result, col='blue', pch=16, main=' Graphe des valeurs prédites')
abline(0,1)

####-3.8- Error * RMSE *:
RMSE.NN = (sum(( data_test_labels- pr.nn)^2) / nrow(data_test)) ^ 0.5
cat('The mean error of the sums of squares:',RMSE.NN)

###-4- Part 02: Application of the * keras * method : 
####-4.1-Data construction:

#Loading the library
library(keras)
X_train <- array_reshape(data_train, c(nrow(data_train),2308))
X_test <- array_reshape(data_test, c(nrow(data_test), 2308))

####-4.2-Recoding of the target variable Y_train and Y_test:

Y_train <- to_categorical(data_train_labels, 2)
Y_test <- to_categorical(data_test_labels, 2)

####-4.3- Network structure creation : 

#Initial model
model <- keras_model_sequential() 
#Layer connecting input and output
model %>% 
  layer_dense(units = 150, activation = 'relu', input_shape = c(2308)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 2, activation = 'softmax')
# Model fit 
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)
history <- model %>% fit(
  X_train, Y_train, 
  epochs = 30, batch_size =256, 
  validation_split = 0.1
)

####-4.4- History of evolution of loss and accuracy values : 
plot(history)

####-4.5- Evaluation of the model:

#Loss rate and accuracy:
model %>% evaluate(X_test, Y_test)
model %>% predict_classes(X_test)
pred<-model %>% predict_classes(X_test)

####-4.6- The confusion matrix:
library(caret)
confusionMatrix(as.factor(data_test_labels),as.factor(pred))


####-4.7- *RMSE* Error:
RMSE.keras= (sum(( data_test_labels- pred)^2) / nrow(data_test)) ^ 0.5
cat('The mean error of the sums of squares:',RMSE.keras)
