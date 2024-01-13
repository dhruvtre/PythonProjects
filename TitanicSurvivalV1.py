#importing all relevant packages and libraries
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

#one-hot encoding categorical inputs to enter into tensorflow
def prepare_categorical_inputs(data):
  OHE = OneHotEncoder()
  OHE.fit(data)
  data_enc = OHE.transform(data)
  return data_enc, OHE

#reading the train data
titanic_train = pd.read_csv("train.csv")

#Creating a Features and Target Data Set
Features = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
Target = 'Survived'
Features_train = titanic_train[Features]
Target_train = titanic_train[Target]

#identifying the empty values in Features_Train
missing_values_in_features = Features_train.isna().sum()
print("Missing values in Features_train:")
print(missing_values_in_features)

#identifying the empty values in Target_train
missing_values_in_target = Target_train.isna().sum()
print("\nMissing values in Target_train:", missing_values_in_target)

#dropping Age and Cabin since irrelevant for the project and too many empty values
Features_train = Features_train.drop(['Age', 'Cabin'], axis=1)

#dropping Name column because irrelevant for the project
Features_train = Features_train.drop('Name', axis=1)

#Filling up the empty values in the Embarked column
Features_train['Embarked'] = Features_train['Embarked'].fillna('S')

#printing the empty values and the dtypes for review

#identifying the empty values in Features_Train
missing_values_in_features = Features_train.isna().sum()
print("Missing values in Features_train:")
print(missing_values_in_features)

#identifying the empty values in Target_train
missing_values_in_target = Target_train.isna().sum()
print("\nMissing values in Target_train:", missing_values_in_target)

#printing the data type
print(Features_train.dtypes)
print(Target_train.dtypes)

#turning sex into categorical data

#taking sex out and turning it into a data frame of its own
sex_updated = Features_train['Sex'].values.reshape(-1, 1)

#passing the updated data frame through the One-Hot Encoding function and outputting sex_enc
sex_enc, OHE = prepare_categorical_inputs(sex_updated)

#One-Hot Encoding outputs a CSR Matrix or something so turning it into a dataframe and also getting column names from encoding key
sex_enc_df = pd.DataFrame(sex_enc.toarray(), columns=OHE.get_feature_names_out())

#concatinating the encoded sex df to the Features_train df
Features_train = pd.concat([Features_train, sex_enc_df], axis=1)

#printing the Features_train head for review
print(Features_train.head())

embarked_updated = Features_train['Embarked'].values.reshape(-1,1)
embarked_enc, OHE = prepare_categorical_inputs(embarked_updated)
embarked_enc_df = pd.DataFrame(embarked_enc.toarray(), columns=OHE.get_feature_names_out())
Features_train = pd.concat([Features_train, embarked_enc_df], axis=1)
print(Features_train.head())

Features_train = Features_train.drop('Ticket', axis=1)
Features_train = Features_train.drop('Embarked', axis=1)
Features_train = Features_train.drop('Sex', axis=1)

missing_values_in_features = Features_train.isna().sum()
print("Missing values in Features_train:")
print(missing_values_in_features)

# Since Target_train is likely a Series, you can directly apply isna() and sum()
missing_values_in_target = Target_train.isna().sum()
print("\nMissing values in Target_train:", missing_values_in_target)
print(Features_train.dtypes)
print(Target_train.dtypes)

from sklearn.model_selection import train_test_split

#splitting the training data into training and validation sets
Features_train1, Features_val, Target_train1, Target_val = train_test_split(Features_train, Target_train, test_size=0.2, random_state=42)
print(Features_train1)
print(Features_val)
print("The type of Features_train/val is", type(Features_train1))

#converting the pandas df into numpy array to feed into the model and reviewing its shape
Features_train1_processed = Features_train1.values
print(Features_train1_processed.shape)
print("The type of Features_train1_processed is", type(Features_train1_processed))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(96, activation='relu', input_shape=(Features_train1_processed.shape[1],)),
    tf.keras.layers.Dropout(0.3),  # Optional: Dropout layer to reduce overfitting
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

pred = model(Features_val.values)
print(pred)

history = model.fit(Features_train1_processed, Target_train1.values, epochs=150, batch_size=128, validation_data=(Features_val.values, Target_val.values))

training_loss = history.history['loss']
validation_loss = history.history['val_loss']
print(type(training_loss), type(validation_loss))

import matplotlib.pyplot as plt

epochs = range(1, len(training_loss) + 1)

plt.plot(epochs, training_loss, 'bo', label='Training loss')
plt.plot(epochs, validation_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

pred = model(Features_val.values)
print(pred)

import numpy as np

# Assuming 'pred' contains the output probabilities from the model
binary_predictions = np.where(pred >= 0.5, 1, 0)

print(binary_predictions)
print(Target_val.values)
print(type(Target_val.values))
print(type(binary_predictions))
print(binary_predictions)

print(len(Target_val.values))
print(len(binary_predictions))



# Check if all elements are equal
all_equal = np.all(Target_val.values == binary_predictions)
print("Are all elements equal?:", all_equal)

# Check if any elements are equal
any_equal = np.any(Target_val.values == binary_predictions)
print("Are any elements equal?:", any_equal)

matching_values_count = np.sum(Target_val.values == binary_predictions.flatten())
print(matching_values_count)

print("Shape of Target_val.values:", Target_val.values.shape)
print("Shape of binary_predictions:", binary_predictions.shape)

# Ensure that the data types are the same
print("Data type of Target_val.values:", Target_val.values.dtype)
print("Data type of binary_predictions:", binary_predictions.dtype)

#reading the test data
test_df = pd.read_csv("test.csv")

#printing the head and length and missing values to analyse the data
print(test_df.head())
print(len(test_df))
print(test_df.isna().sum())

#getting rid of the age and the cabin columns because of the empty values
test_df = test_df.drop(['Age', 'Cabin', 'Name'], axis=1)

#printing data type for each column to start converting it into int etc to pass into the model
print(test_df.dtypes)


#reshaping all columns into 1d numpy 
Embarked_test_updated = test_df['Embarked'].values.reshape(-1,1)
Sex_test_updated = test_df['Sex'].values.reshape(-1, 1)
Ticket_test_updated = test_df['Ticket'].values.reshape(-1, 1)

#passing each object dtype through OHE to come back as float/int
Sex_test_enc, OHE1 = prepare_categorical_inputs(Sex_test_updated)
Embarked_test_enc, OHE2 = prepare_categorical_inputs(Embarked_test_updated)
Ticket_test_enc, OHE3 = prepare_categorical_inputs(Ticket_test_updated)
print(type(Sex_test_enc), type(Embarked_test_enc), type(Ticket_test_enc))
print(Sex_test_enc)

#turning all scipy matrices into dfs to be concatenated
Sex_test_enc_df = pd.DataFrame(Sex_test_enc.toarray(), columns=OHE1.get_feature_names_out())
Embarked_test_enc_df = pd.DataFrame(Embarked_test_enc.toarray(), columns=OHE2.get_feature_names_out())
Ticket_test_enc_df = pd.DataFrame(Ticket_test_enc.toarray(), columns=OHE3.get_feature_names_out())

#printing the head of each encoded df for review
print(Sex_test_enc_df.head(), Embarked_test_enc_df.head(), Ticket_test_enc_df.head())

#getting rid of ticket from the original test_df and concatenating the Sex and Embarked encoded columns to it
test_df = pd.concat([test_df, Sex_test_enc_df, Embarked_test_enc_df], axis=1)

#printing the new head of test_df
print(test_df.head())

#dropping the original sex, ticket, and embarked columns
test_df = test_df.drop(['Sex', 'Ticket', 'Embarked'], axis=1)

#printing the type of each column in the df now 
for column in test_df:
  print(test_df[column].dtype)

#converting the pandas df into numpy array to feed into the model and reviewing its shape
test_data_processed = test_df.values
print(test_data_processed.shape)
print("The type of test_data_processed is", type(test_data_processed))

#using the model on the test data
pred_final = model(test_data_processed)

#checking the output file 
print(type(pred_final))
print(pred_final.shape)
print(pred_final)

binary_predictions_test = np.where(pred_final >= 0.5, 1, 0)
print(binary_predictions_test)

#converting binary predicitons into df

test_pred_final = pd.DataFrame(binary_predictions_test, columns=['Survived'])
#concatenating the binary predictions to the test df
test_df = pd.concat([test_df, test_pred_final], axis=1)

#printing test_df head
print(test_df.head())

#getting rid of all columns except passenger id and prediction and printing the shape of the df as well the dtypes
test_df = test_df.drop(['Pclass', 'SibSp', 'Parch', 'Fare', 'x0_female', 'x0_male', 'x0_C', 'x0_Q', 'x0_S'], axis=1)

print(test_df.head())
print(test_df.shape)
print(test_df.head)

file_path = '/content/test_df.csv'
test_df.to_csv(file_path, index=False)

file_path
