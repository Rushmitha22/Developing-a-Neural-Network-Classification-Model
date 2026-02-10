# Experiment 2 : Developing a Neural Network Classification Model
## NAME : RUSHMITHA  R
## REGISTRATION NUMBER : 212224040281

## AIM :
To develop a neural network classification model for the given dataset.

## THEORY :
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model:
<img width="1097" height="940" alt="NN DIAGRAM EXP 2" src="https://github.com/user-attachments/assets/534ffd4a-00c9-4ca4-84a0-383308dc9a37" />


## DESIGN STEPS
### STEP 1: 
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2: 
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.


### STEP 3: 
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.


### STEP 4: 

Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5: 

Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6: 
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.




## PROGRAM:

### Name: Rushmitha R

### Register Number: 212224040281

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)


    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs,labels in train_loader:
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

train_model(model,train_loader,criterion,optimizer,epochs=100)
```

### Dataset Information
<img width="1287" height="268" alt="dataset exp2" src="https://github.com/user-attachments/assets/1b6a3580-4764-4f9f-8a3d-4be107a09c2e" />

### OUTPUT
<img width="460" height="56" alt="output exp2" src="https://github.com/user-attachments/assets/32729937-1ad8-42b5-8a12-de6080e5e7fc" />

## Confusion Matrix
<img width="676" height="572" alt="confusion exp 2" src="https://github.com/user-attachments/assets/6e528d33-1839-4285-b5e1-791b0cf27718" />



## Classification Report
<img width="627" height="442" alt="calssification exp 2" src="https://github.com/user-attachments/assets/a58c2f29-8252-4259-b1fa-d18ea8c116d5" />

### New Sample Data Prediction

<img width="401" height="115" alt="2" src="https://github.com/user-attachments/assets/50e911f5-efaa-42fe-bb14-7e844125f172" />

## RESULT
This program has been executed successfully.
