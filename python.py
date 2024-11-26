  # hello wrold this is mezbah from backend developer 
  # lets use our blueprint and build this projects 
  # lets do this with proper code ....

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, time, functools
import pathlib as Path

# Let's start with the basic code.
# This is our first class -->

class FirstCls:
    """This is dataset's class"""
    def __init__(self, data, ai_data):
        self.data = data
        self.ai_data = ai_data

    @staticmethod
    def warning_call():
        print("The system has encountered an error!")

    @staticmethod
    def welcome_call():
        user_calls = str(input('wanna start the sever (Y/N): '))
        if user_calls.endswith("y"):
            print("Opening server...")

        elif user_calls.endswith("n",'N') :
            exit()
            
    @staticmethod
    def closing_call(instance):
        """Handles the closing process. If the user doesn't want to close, the system should
    return to the main access point without breaking the model's functionality."""
        while True:
            user_call = input("Wanna close the system? (y/n): ")
            if user_call.lower() == "y":
                print("Closing system...")
                break  # Exit the loop after closing
            elif user_call.lower() == "n":
                if instance is not None:
                    print("Returning to the main access point...")
                    instance.get_user_access()
                else:
                    print("Error: No valid instance provided to return to the main access point.")
                    break 
            else: 
                print("Invalid input. Please enter 'y' or 'n'.")

    def humanoid_dataset(self):
        try:
            humanoid_data = {"Humanoid data": self.data}
            return humanoid_data
        except ValueError:
            return self.warning_call()

    def ai_dataset(self):
        try:
            ai_dataset = {"AI dataset": self.ai_data}
            return ai_dataset
        except ValueError:
            return self.warning_call()


# This is our functions class -->
class SecondCls(FirstCls):
    """This is the functions class"""
    def __init__(self, data, ai_data):
        super().__init__(data, ai_data)

    def first_functions(self):
        print("Hey there, let's prepare your dataset\n")
        data_load_choice = int(input("Enter your dataset format: 1 for XLSX or 2 for CSV: "))
        try:
            if data_load_choice == 1:
                file_path = input("Enter your XLSX file path: ")
                data = pd.read_excel(file_path)
            elif data_load_choice == 2:
                file_path = input("Enter your CSV file path: ")
                data = pd.read_csv(file_path)
            print(data)
        except (TypeError, ImportError, FileNotFoundError):
            return self.warning_call()

        print("\n1. Check data")
        print("\n2. Perform actions on data\n")
        main_step_choice = int(input("Enter your choice: "))
        
        if main_step_choice == 1:
            self.data_inspection_options(data)
        elif main_step_choice == 2: 
            self. data_update_options(data)

    def data_inspection_options(self, data):
        print("1. Read data in head/tail/middle rows or columns")
        print("2. Check data null values")
        print("3. Check data non-null values")
        print("4. Describe data")
        print("5. Show data info and structure")
        print("6. Display data graph with columns and rows")
        print("7. Show unique values in data")
        print("8. Show data items")
        print("9. Show numerical sum of columns")
        print("10. Display specific rows and columns")

        sub_choice_01 = int(input("Enter your choice: "))
        if sub_choice_01 == 1:
            self.display_rows(data)
        elif sub_choice_01 == 2:
            self.check_nulls(data)
        elif sub_choice_01 == 3:
            self.check_non_nulls(data)
        elif sub_choice_01 == 4: 
            self.data_describe(data)
        elif sub_choice_01 == 5: 
            self.data_info(data)
        elif sub_choice_01 == 6: 
            self.data_graph(data)
        elif sub_choice_01 == 7: 
            self.unique_value(data)
        elif sub_choice_01 == 8: 
            self.data_items(data)
        elif sub_choice_01 == 9: 
            self.columns_avg_med_low(data)
        elif sub_choice_01 == 10: 
            self.view_rows_columns(data)

    def display_rows(self, data):
        ask = int(input('1. Head / 2. Tail / 3. Middle: ')) 
        if ask == 1:
            num1 = int(input('Enter number of rows to display from head: '))
            print(data.head(num1))
        elif ask == 2:
            num2 = int(input('Enter number of rows to display from tail: '))
            print(data.tail(num2))
        elif ask == 3:
            num3 = int(input('Enter number of rows to display from middle: '))
            middle = len(data) // 2
            print(data.iloc[middle - num3 // 2: middle + num3 // 2])

    def check_nulls(self, data):
        print("1. Total count of null values")
        print("2. Null values by column")
        null_choice = int(input("Enter your choice: "))

        if null_choice == 1:
            print("Total null values:", data.isnull().sum().sum())
        elif null_choice == 2:
            print("Null values by column:\n", data.isnull().sum())

    def check_non_nulls(self, data):
        print("1. Total count of non-null values")
        print("2. Non-null values by column")
        not_null_choice = int(input("Enter your choice: "))

        if not_null_choice == 1:
            print("Total non-null values:", data.notnull().sum().sum())
        elif not_null_choice == 2:
            print("Non-null values by column:\n", data.notnull().sum())

    def data_describe(self, data):
        print('Describing the data...')
        print(data.describe())
    
    def data_info(self, data):
        print('Dataset information:')
        print(data.info())

    def data_graph(self, data):
        graph_choice = int(input('Choose: 1 for Seaborn, 2 for Matplotlib :  '))
        column_name = input('Enter the column name for the graph: ')
        
        if graph_choice == 1:
            sns.boxplot(x=column_name, data=data)
        elif graph_choice == 2:
            plt.boxplot(data[column_name])
        else:
            return self.warning_call()

        plt.title(f'{column_name} column graph')
        plt.grid(True)
        plt.show()
        
    def unique_value(self, data):
        column_name = input("Enter your column name: ")
        if column_name in data.columns:
            print(f"Unique values in '{column_name}':\n{data[column_name].unique()}")
        else:
            print(f"Column '{column_name}' not found.")

    def data_items(self, data):
        print("Dataset items:")
        for column, series in data.items():
            print(f"\nColumn '{column}':")
            print(series.values)

    def columns_avg_med_low(self, data):
        input_data_columns = input('Enter your data columns (comma-separated): ')
        columns = [col.strip() for col in input_data_columns.split(',')]  # Splitting input and stripping whitespace
        valid_columns = [col for col in columns if col in data.columns]
        if not valid_columns:
            print("No valid columns found in the dataset.")
        for col in valid_columns:
            try:
                avg_value = data[col].mean()
                median_value = data[col].median()
                low_value = data[col].min()
                print(f"Column: {col}")
                print(f"  Average: {avg_value}")
                print(f"  Median: {median_value}")
                print(f"  Low value: {low_value}")
            except Exception as e:
                print(f"Error calculating values for column {col}: {e}")


    def view_rows_columns(self, data):
        column_name = input("Enter your column name: ")
        if column_name in data.columns:
            print(f"Values in column '{column_name}':\n{data[column_name]}")
        else:
            print(f"Column '{column_name}' not found.")
            
    
    def data_update_options(self, data):
        print('1. Remove the duplicates from the overall dataset')
        print('2. Remove the duplicates from specific rows')
        print('3. Remove the duplicates from specific columns')
        print('4. Fill the NaN values in the overall dataset')
        print('5. Fill the NaN values in specific rows')
        print('6. Fill the NaN values in specific columns')
        print('7. Find outliers in the overall dataset')
        print('8. Find outliers in specific columns')
        print('9. Remove outliers in the overall dataset')
        print('10. Remove outliers in specific columns')
        print('11. Encode the overall dataset')
        print('12. Encode specific columns')
        print('13. Use feature scaling in specific columns')
        print('14. Use feature scaling in the overall dataset')
        print('15. Use train_test_split on specific columns')

        sub_choice_001 = int(input("Enter your choice: "))

        if sub_choice_001 == 1:
            data = self.remove_all_duplicates(data)
        elif sub_choice_001 == 2:
            data = self.remove_duplicates_specific_rows(data)
        elif sub_choice_001 == 3:
            data = self.remove_duplicates_specific_columns(data)
        elif sub_choice_001 == 4:
            data = self.fill_NaN_overall(data)
        elif sub_choice_001 == 5:
            data = self.fill_NaN_specific_rows(data)
        elif sub_choice_001 == 6:
            data = self.fill_NaN_specific_columns(data)
        elif sub_choice_001 == 7:
            data = self.find_outliers_overall(data)
        elif sub_choice_001 == 8:
            data = self.find_outliers_specific_columns(data)
        elif sub_choice_001 == 9:
            data = self.remove_outliers_overall(data)
        elif sub_choice_001 == 10:
            data = self.remove_outliers_specific_columns(data)
        elif sub_choice_001 == 11:
            data = self.encode_overall_dataset(data)
        elif sub_choice_001 == 12:
            data = self.encode_specific_columns(data)
        elif sub_choice_001 == 13:
            data = self.feature_scaling_specific_columns(data)
        elif sub_choice_001 == 14:
            data = self.feature_scaling_overall_dataset(data)
        elif sub_choice_001 == 15:
            data = self.train_test_split_specific_columns(data)


    def remove_all_duplicates(self, data):
        print('Your request is processing ...')
        data = data.drop_duplicates()
        print('Duplicates removed from the dataset.') 
        data.to_csv('new_data.csv', index=False)
        with open('new_data.csv', 'r') as file:
            print(file.read())
            return data


    def remove_duplicates_specific_rows(self, data):
        specific_rows = list(map(int, input("Enter row indices to check for duplicates (comma-separated): ").split(',')))
        data = data.iloc[specific_rows].drop_duplicates()
        print("Duplicates removed from specified rows.")
        return data
    
    def remove_duplicates_specific_columns(self, data):
        columns = input("Enter column names to check for duplicates (comma-separated): ").split(',')
        for col in columns:
            if col in data.columns:
                data = data.drop_duplicates(subset=col)
                print(f"Duplicates removed from column: {col}.")
        return data

    def fill_NaN_overall(self, data):
        fill_value = input("Enter a value to replace NaN: ")
        data = data.fillna(fill_value)
        print("NaN values replaced in the entire dataset.")
        return data

    def fill_NaN_specific_rows(self, data):
        specific_rows = list(map(int, input("Enter row indices to fill NaN values (comma-separated): ").split(',')))
        fill_value = input("Enter a value to replace NaN: ")
        data.iloc[specific_rows] = data.iloc[specific_rows].fillna(fill_value)
        print("NaN values replaced in the specified rows.")
        return data

    def fill_NaN_specific_columns(self, data):
        columns = input("Enter column names to fill NaN values (comma-separated): ").split(',')
        fill_value = input("Enter a value to replace NaN: ")
        for col in columns:
            if col in data.columns:
                data[col] = data[col].fillna(fill_value)
        print("NaN values replaced in the specified columns.")
        return data

    def find_outliers_overall(self, data):
        z_scores = (data - data.mean()) / data.std()
        outliers = z_scores[(z_scores > 3).any(axis=1)]
        print("Outliers found:\n", outliers)
        return data

    def find_outliers_specific_columns(self, data):
        column = input("Enter the column to find outliers: ")
        if column in data.columns:
            z_scores = (data[column] - data[column].mean()) / data[column].std()
            outliers = data[abs(z_scores) > 3]
            print("Outliers found in the column:\n", outliers)
        return data

    def remove_outliers_overall(self, data):
        z_scores = (data - data.mean()) / data.std()
        data = data[(abs(z_scores) <= 3).all(axis=1)]
        print("Outliers removed from the dataset.")
        return data

    def remove_outliers_specific_columns(self, data):
        column = input("Enter the column to remove outliers: ")
        if column in data.columns:
            z_scores = (data[column] - data[column].mean()) / data[column].std()
            data = data[abs(z_scores) <= 3]
            print(f"Outliers removed from the column: {column}.")
        return data

    def encode_overall_dataset(self, data):
        data = pd.get_dummies(data)
        print("Dataset encoded.")
        return data

    def encode_specific_columns(self, data):
        columns = input("Enter column names to encode (comma-separated): ").split(',')
        for col in columns:
            if col in data.columns:
                data[col] = pd.factorize(data[col])[0]
                print(f"Column {col} encoded.")
        return data

    def feature_scaling_specific_columns(self, data):
        columns = input("Enter column names to scale (comma-separated): ").split(',')
        for col in columns:
            if col in data.columns:
                data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
                print(f"Feature scaling applied to column: {col}.")
        return data

    def feature_scaling_overall_dataset(self, data):
        data = (data - data.min()) / (data.max() - data.min())
        print("Feature scaling applied to the entire dataset.")
        return data

    def train_test_split_specific_columns(self, data):
        from sklearn.model_selection import train_test_split
        columns = input("Enter column names to use for train-test split (comma-separated): ").split(',')
        if all(col in data.columns for col in columns):
            X_train, X_test = train_test_split(data[columns], test_size=0.2, random_state=42)
            print("Train-test split applied. Train and test sets created.")
        else:
            print("Invalid choice!")
        return data


    def second_functions(self) : 
        print('Lets prepaired the dataset with Ai actions ')
        data_load_choice_ai = int(input('Enter youre dataset as (1.csv / 2.xlsx) : '))
        try : 
            if data_load_choice_ai == 1:
                file_path_ai = input('Enter your csv file path : ')
                data_ai = pd.read_csv(file_path_ai)
            elif data_load_choice_ai == 2:
                file_path_ai = input('Enter your xlsx file path : ')
                data_ai = pd.read_excel(file_path_ai)
            print(data_ai)
            print('Now, let\'s apply Ai actions on the dataset')
        except ImportError or TypeError : 
            return self.warning_call()
        
    def starting_call (self , data ) : 
        print('Welcome to Data Management System')
        user_acresss_call = input('wanna start the process (y/n): ')
        try : 
            if user_acresss_call.endswith("y"):
                self . proceess_actions (data)
            elif user_acresss_call .endswith('n') : 
                self.closing_call ()
            else : 
                EOFError

        except EOFError:
            return self.warning_call()  
        

    def  proceess_actions (self, data ) : 
        print ('process is started ! ')
        data_nulls_check_ai = data .isnull() .sum() .sum()
        if data_nulls_check_ai is not None : 
            print('NaN Value is founded : ', data_nulls_check_ai)
        








class ThirdCls(SecondCls) :
    """This class will connect layers with functions, dataset, and classes"""
    def __init__(self, data, ai_data) :
        super().__init__(data, ai_data)

    def get_user_access(self):
        print("Want to use Humanoid system (1)")
        print("Want to use AI system (2)")
        choice = int(input("Enter your choice: "))
        try:
            if choice == 1:
                return self.first_functions()
            elif choice == 2:
                return self.second_functions()
            else:
                raise EOFError
        except (ValueError, TypeError):
            return self.warning_call() 
        
   # Lets call All of functions ...

if __name__ == "__main__":
    data_instance = ThirdCls(data="Sample Data", ai_data="Sample AI Data")
    data_instance.welcome_call()
    data_instance.get_user_access()
    data_instance.closing_call(instance=data_instance)  # Pass the actual instance






   