This is a Python program designed to help you predict time series data using deep learning models, specifically using LSTM (Long Short-Term Memory) units. This program works with any dataset that includes time-series data and allows you to customize various aspects of the prediction process.

## Dependencies

-  pip install -r requirements.txt

## Features

- Load your data from a CSV file.
- Select specific time frames for analysis.
- Choose the number of columns to use from your dataset.
- Customize deep learning model settings such as epochs, batch size, and more.
- Reduce memory usage by adjusting data types.
- Validate the dataset to ensure it contains the correct data types.
- Predict future values based on historical data.

3. Download the program files to your local machine.

## Usage

1. Prepare your CSV data file. Ensure it has a datetime index and the columns you wish to analyze.
2. Modify the parameters in the `multi_RNN` class instantiation in the script to match your data and preferences:
- `df`: Path to your CSV file.
- `time_stamp_start`: Start date for the data analysis.
- `time_stamp_end`: End date for the data analysis.
- Other parameters like `epochs`, `batch_size`, `LSTM_units`, etc., as needed.

3. Run the script from your command line:
    python multi_RNN.py

4. Check the output files for predictions and model performance plots.

## Customization

You can customize the program by changing the parameters in the `multi_RNN` class:
- Change `columns_left_in_df` to select how many features (columns) you want to include from your dataset.
- Adjust `float_16`, `float_32`, or `float_64` to manage memory usage depending on your dataset size.
- Modify `LSTM_units` and `length` to tweak the LSTM model configuration.

## Output

The program will output:
- Predictions as CSV files.
- Plots comparing predicted values with actual values.
- Training and validation loss plots to evaluate model performance.
- And a h5 model.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Thank you for using our Time Series Prediction Program! We hope it assists you effectively in your data analysis tasks.
