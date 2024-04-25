import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
import numpy as np
import warnings
import sys
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM


class multi_RNN:
    '''the class that fetching dataframe and validate the data. If we got any
       object, then we throw it back to the user. need to fix it before runing this
       program. only int, uint and float is acceptable. 

       the user can choose which timestamp they want to predict from.
       they can also use an extra feature that creates less memory for the 
       computer to handle if we change the float dtypes to float16 or float32, 
       but if the user wants to have precision in the data i would rather go with 
       float64. test and feel free.
       Columns_left_in_df is a choice for the user to
       minimize the feature, if 2 is choosen you will only get 2 columns and so on.

       epochs 25 is default, the user can choose something else if they want to.
       same with batch_size, activation, lstm_units,length and optimizer.
    '''
    def __init__(self, df: str, time_stamp_start: str,
                 time_stamp_end: str, float_16: bool = False,
                 float_32: bool= False,float_64: bool = False,
                 columns_left_in_df: int = 0, epochs: int = 25, # 0 is all columns.
                 batch_size: int = 1, activation: str = 'tanh',
                 LSTM_units: int = 100, length: int = 1,
                 optimizer: str = 'adam'):
        self.__df = df
        self.csv_name = df
        self.time_stamp_start = time_stamp_start
        self.time_stamp_end = time_stamp_end
        self.float_16 = float_16 if float_16 else float_16
        self.float_32 = float_32 if float_32 else float_32
        self.float_64 = float_64 if float_64 else float_64
        self.columns_left_in_df = columns_left_in_df \
                                if columns_left_in_df \
                                else columns_left_in_df
        self.epochs = epochs if epochs else epochs
        self.batch_size = batch_size if batch_size else batch_size
        self.activation = activation if activation else activation
        self.LSTM_units = LSTM_units if LSTM_units else LSTM_units
        self.length = length if length else length
        self.optimizer = optimizer if optimizer else optimizer
        self.losses = []

    def fetch_Data(self):
        ''' Fetching data from the csv and creating our index.
            We are selecting our dataframe length with timestamp choice
            from the user.

            rounding up to 2 decimals. there will be a choice availible
            to choose float16,float32,float64 to minimize memory usage.
            it's is good if we have a big data and will run all.

            this class also have the choice to select how many columns you
            want to use from the dataset.
            and everything in a try exception to catch the error.
        '''

        self.df_timestamp = pd.read_csv(self.__df, usecols=['date'])
        self.__df = pd.read_csv(self.__df, parse_dates=True, index_col='date')
        self.__df = self.__df.loc[self.time_stamp_start : self.time_stamp_end]
        try:
            self.__df = self.__df.iloc[:,:self.columns_left_in_df]
            self.__df = self.__df.round(2)
            print(self.__df.head())
            print(f"\nData from : {self.time_stamp_start}\n"
                f"Data ends : {self.time_stamp_end}")
            '''this buffer is just to get rid of the None return the df.info does.
               this function i'm using is to get the values from df.info and print
               them as a string.
            '''
            buffer = StringIO()
            self.__df.info(buf=buffer)
            info_str = buffer.getvalue()
            print(f"\n{info_str}\n")

        except ValueError as e:
            print(f"==You got an error==\n{e}")
        for i,dtype in enumerate(self.__df.dtypes):
            column = self.__df.columns[i]
            if dtype == 'float':
                if self.float_16:
                    self.__df[column] = self.__df[column].astype('float16')
                elif self.float_32:
                    self.__df[column] = self.__df[column].astype('float32')
                elif self.float_64:
                    self.__df[column] = self.__df[column].astype('float64')
                else:
                    print("\nUsing orignal type values\n")
        
     
    
    def validate_dataframe(self):
        ''' Validation method thats checks if we got anything else than
            integers,undefined integers or float in the dataset. 
            if we do so we have a list comprehension that will catch
            the name of the column and print it out. it gets easier for 
            the user to quick do a check up on that column.
        '''
        try:
            check_for_bad = [col for col in self.__df\
                             if self.__df[col].dtype.kind not in 'iuf']
            if check_for_bad:
                raise ValueError(f"The Columns that you need to check are : "
                                 f"{check_for_bad}\n"+self.csv_name+" contains"
                                  " non-numeric values\n")
        except KeyError:
            raise ValueError(check_for_bad+" column not found in dataframe\n")
        else:
            print("--Dataframe validation successful--\n")

    def create_DataFrames_w_one_column(self):
        '''Creating multiple dataframes with timestamp index and one column
           per each DataFrame. Saving the column names with the return so we 
           can use them in other methods as well.
        '''
        self.all_dataframes = []
        all_column_names = []
        for column in self.__df.columns:
            all_column_names.append(column)
            onecolumn = pd.DataFrame(data = \
                                     self.__df[column],index= self.__df.index)
            self.all_dataframes.append(onecolumn)
            try:
                onecolumn.to_csv("multi_RNN_"+column+".csv")
            except Exception as e:
                print(f"MESSAGE FOR YOU {e}")
            finally:
                #printing out no matter what.
                print(f"--CREATED [{column}] DATAFRAME--")
        return all_column_names

    def build_model_per_column(self,one_column_df):
        ''' This method creates train and test for the generators.
            train and test splits with test_index for 2 days * length.
            early stopage we can stop with patience. that creates 
            a more secure model to resist overfitting and underfitting.
            low batch_size will lead to more training time, but will
            contribute to better generalization by exposing the model
            to more variation and diversity in the training data.  

        '''
       
        
        test_days = 2
        test_index = test_days * self.length
        n_features = len(one_column_df.columns)
        early_stop =  EarlyStopping(       monitor = 'val_loss',
                                           mode = 'auto',
                                           patience = 2
                                   )
        
        train = one_column_df[test_index:]
        test = one_column_df[:test_index]
        
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train)
        scaled_test = scaler.transform(test)
        
        train_generator = TimeseriesGenerator(scaled_train,
                                                scaled_train,
                                                length=self.length,
                                                batch_size = self.batch_size)
        test_generator = TimeseriesGenerator(   scaled_test,
                                                scaled_test,
                                                length = self.length,
                                                batch_size = self.batch_size)
    
        model = Sequential()
        model.add(LSTM( units = self.LSTM_units,
                        activation = self.activation,
                        input_shape = (self.length,n_features)))
        model.add(Dense(n_features))

        model.compile(optimizer = self.optimizer, loss = 'mse')

        result = model.fit(train_generator,
                            epochs = self.epochs,
                            validation_data = test_generator,
                            callbacks = [early_stop],
                            use_multiprocessing = True)
        save_model(model, one_column_df.columns[0]+'.h5')
        losses = pd.DataFrame(result.history)
        
        return losses, test_generator, test, scaler
    
    def build_single_model_per_column(self):
        ''' this method looping a method that creates model for each
            single dataframes. and appending the losses,test_generator,
            test_df and the scaler. so we can use it later in plotting and 
            in the predict methods.

            returning only losses just to seperate the classes from each other.
            I created an extra class for plotting that doesnt inheriete from 
            eachother, just using losses variabel.
        '''
        self.all_test_gen = []
        self.all_test_df = []
        self.scaler = []
        for i,each_df_column in enumerate(self.all_dataframes):
            print(f"\nNR {i+1} ||"
                  f" [DATAFRAME NAME: {each_df_column.columns[0]}]")
            losses,test_gen, test_df,scaler = \
                            self.build_model_per_column(each_df_column)
            self.losses.append(losses)
            self.all_test_gen.append(test_gen)
            self.all_test_df.append(test_df)
            self.scaler.append(scaler)
       
        return self.losses
    def predict_machine(self):
        '''predict method that predict single dataframe at a time.
           the method loads in models from your directory and uses the scaler
           that we saved earlier for each dataframe. inverse_transform.reshape 
           to get the values to original values and not scaled.

           creating a new dataframe and csv with the original values and 
           predicted together using concat.
        '''
        for i, each_df_column in enumerate(self.all_test_df):
            try:
                column = each_df_column[:len(self.all_test_gen[i])]
                timestamp_index = column.index
                loaded_model = load_model(each_df_column.columns[0]+".h5")
                y_pred = loaded_model.predict(self.all_test_gen[i])
                y_pred = self.scaler[i].inverse_transform(y_pred.reshape(-1, 1))
                pred_df = pd.DataFrame(data=y_pred, columns=['PREDICTION'],
                                        index=timestamp_index)
                orginal_df_with_pred = pd.concat([column, pred_df], axis=1)
                orginal_df_with_pred.to_csv(each_df_column.columns[0]+\
                                                    "_predictions.csv")
            except Exception as e:
                print(f"[i] Check the error and repair it [i]\n{e}")
            finally:
                #printing out no matter what.
                print(f"--CREATED [{column}] DATAFRAME--")
                
    def predict_one_line(self):
        '''This method will predict the first line that i choosed.
           and also printing it out to show the user the result from
           original value and the predicted.
           loading the models here as well. to make sure we are using 
           the correct model.
           creating a timestamp on the predicted single row as well
           to get a clearer view of the data.
        '''
        predict_data = []
        single_row = self.__df.iloc[:1,:self.columns_left_in_df]
        
        for num in range(len(single_row.columns)):
            try:
                scaller = MinMaxScaler()
                loaded_model = load_model(self.__df.columns[num] + ".h5")
                test = np.array([single_row.iloc[0,num]]).reshape(-1,1)
                pred = self.scaler[num].transform(test)
                pred = loaded_model.predict(pred)
                pred = self.scaler[num].inverse_transform(pred)
                predict_data.append(pred)
            except Exception as e:
                print(f'[i] Check the error and repair it [i]\n{e}')
        
        pred_DD = pd.DataFrame(data=np.array(\
                          predict_data).reshape(-1,len(predict_data))\
                        , columns= single_row.columns)
        pred_DD['date'] = single_row.index[0]
        pred_DD = pred_DD.set_index('date')
        print(single_row.head())
        print(pred_DD.head())

class plotmajster:
    ''' A new class just to seperate them two from what
       they are suppose to do. i just really want to see if
       this class creation would make it more clearly for other
       developers to read the code. i just want to develop as a 
       developer my self and see what's better or not. 

       This class just plot everything.

    '''
    def __init__(self,column_names):
        self.__column_names = column_names
        
    def plot_predict(self):
        ''' reading in the csv file that we made earlier.
            to plot the original vs prediction, and later saving
            it to png.
        '''
        count = 0
        for column in self.__column_names:
            df = pd.read_csv(column+'_predictions.csv')

            fig, ax = plt.subplots(figsize=(10, 6))

            df.plot(ax=ax)
            plt.xlabel('X-Trend')
            plt.ylabel('Y-Trend')
            plt.title(column+' Test Values vs Predicted Values', fontsize = 18)

            plt.savefig(column+'_vs_predictions.png')
            plt.show(block = True)
            count +=1 

    def plot_losses(self,list_w_losses):
        ''' plotting the training loss vs validation loss.
            and saving it to a png file and csv.
        '''
        count = 0
        for train_loss_val_loss in list_w_losses:
            training_loss = train_loss_val_loss['loss']
            validation_loss = train_loss_val_loss['val_loss']
            epochs = range(1, len(training_loss) + 1)
            plt.plot(epochs, training_loss, 'b', label='Training Loss')
            plt.plot(epochs, validation_loss, 'r', label='Validation Loss')
            plt.title('Training Loss and Validation Loss', loc='left')
            plt.title(self.__column_names[count],loc='right',fontsize = 22)
            plt.xlabel('Epochs')
            plt.ylabel('Loss-Trend')
            plt.savefig(self.__column_names[count]+'_loss_val_loss.png')
            plt.legend()
            plt.show()
            train_loss_val_loss.to_csv(self.__column_names[count]+'_losses.csv')
            count += 1 

'''column_left_in_df is a choice for the user, if they want
   just run the program with just the 2 first columns or more.
   if you choose 0 it will run all columns. and 0 is default.

   i have done this app most for the energydata_complete.csv,
   but you can run it with the RSCCASN.csv, but need to change the 
   timestamp to 1992-01-01 to 2019-10-01 for example and lstm_units,length.
'''
MY_RRN = multi_RNN( df = 'energydata_complete.csv',
                    time_stamp_start='2016-01-11',
                    time_stamp_end='2016-01-17',
                    columns_left_in_df = 2,
                    LSTM_units=144,
                    batch_size=1,
                    epochs=2,
                    activation='tanh',
                    length=144,
                    float_16 = False,
                    float_32 = False,
                    float_64 = False)


def main():
    '''Main method.
    '''
    warnings.filterwarnings("ignore")
    try:
        MY_RRN.fetch_Data()
        MY_RRN.validate_dataframe()
        all_column_names = MY_RRN.create_DataFrames_w_one_column()
        losses = MY_RRN.build_single_model_per_column()
        MY_RRN.predict_machine()
        MY_PLOT = plotmajster(all_column_names)
        MY_PLOT.plot_predict()
        MY_PLOT.plot_losses(losses)
        MY_RRN.predict_one_line()
    except Exception as e:
        print(e)
        sys.exit('[i] Check the error and repair it [i]')


if __name__ == '__main__':
    main()
    sys.exit("--Thank you for using this program--")

#By Rasmus Albertsson
