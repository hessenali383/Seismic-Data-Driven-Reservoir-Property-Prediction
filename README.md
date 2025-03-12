# Seismic Data-Driven Reservoir Property Prediction

**Project Overview**

This project focuses on leveraging machine learning (ML) techniques to predict key reservoir properties, specifically porosity and volume of shale (Vshale), using seismic data. The ultimate goal is to integrate these predictions into dynamic reservoir simulations to obtain a 3D representation of these properties within the reservoir. This approach enables more accurate reservoir characterization and improved decision-making in hydrocarbon exploration and production.

**Data Source**

The data used in this project originates from the F3 Block, located offshore in the North Sea, Netherlands. The dataset is publicly available and can be accessed through the following source:

*   **Data Source:** [TerraNubis F3 Demo 2020](https://terranubis.com/datainfo/F3-Demo-2020)
*   **Country:** Netherlands
*   **Location:** Offshore, North Sea
*   **Blocks:** F3
*   **Coordinates:** N 54° 52’ 0.86” / E 4° 48’ 47.07”
* **Well Log Data** one well was used.

**Workflow**

The project follows a structured workflow that integrates seismic data processing, feature extraction, machine learning, and prediction:

1.  **Data Acquisition and Preprocessing:**
    *   Seismic and well log data were acquired from the F3 Block dataset.
    *   One well's data was used.
    *   The data was loaded into the OpendTect software (a seismic interpretation platform).

2.  **Seismic Trace Extraction:**
    *   Seismic trace data was extracted from OpendTect in CSV format. This format is suitable for loading into Python-based ML libraries.

3.  **Feature Engineering:**
    *   A range of seismic attributes were computed, potentially including:
        *   Instantaneous Amplitude
        *   Instantaneous Frequency
        *   Instantaneous Phase
        *   Hilbert Attribute
        *   Envelope with Phase
        *   Envelope with Frequency
        *   Similarity
        *   Volume Statistics
        *   Spectral Decomposition
        *   Frequency Average
        *   Energy
        *   Semblance
        *   Texture
        *   AVO (Amplitude Variation with Offset) - related attributes (e.g., AVO Porosity, AVO Lithology)
        *   Relief
        *   F3 Seismic Data (raw seismic amplitudes)
        *   F3 Velocity Data

4.  **Machine Learning Model Building:**
    *   Three ML models were developed and evaluated:
        *   **Linear Regression:** A baseline model for comparison.
        *   **Random Forest Regressor:** An ensemble method known for its robustness and ability to handle non-linear relationships.
        *   **Neural Network (MLPRegressor):** A multi-layer perceptron regressor for capturing complex patterns.
    *   The dataset was split into training (75%) and testing (25%) sets.
    *   A `StandardScaler` was used to standardize the features (important for neural networks).

5.  **Model Training and Evaluation:**
    *   Models were trained on the training data.
    *   Model performance was evaluated using:
        *   **Root Mean Squared Error (RMSE):**  Measures the average magnitude of the errors. Lower RMSE is better.
        *   **R-squared (R2) Score:**  Represents the proportion of variance in the dependent variable (Vshale) that is predictable from the independent variables (seismic attributes). Higher R2 is better (closer to 1).

6.  **3D Property Prediction:**
    *   SEG-Y files were prepared, containing the seismic attributes as input.  *Important Note:* The code for generating these SEG-Y files is not included in the provided notebook, but it is a crucial part of the workflow for 3D prediction.
    *   The trained ML models (specifically the best-performing one, the Neural Network) were used within OpendTect to predict porosity and Vshale in 3D throughout the seismic volume.

**CSV Data Structure**

The CSV file (e.g., `f3_volume_of_shale.csv`) contains the following columns:

| Column Name              | Description                                                                                                                                | Data Type |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | --------- |
| Time                     | Two-way travel time (related to depth)                                                                                                    | float     |
| Volume\_of\_Shale         | Target variable: Volume of shale (Vshale) - the property to be predicted.                                                                 | float     |
| Inst\_Amp                | Instantaneous Amplitude                                                                                                                    | float     |
| Inst\_Freq               | Instantaneous Frequency                                                                                                                    | float     |
| Inst\_Phase              | Instantaneous Phase                                                                                                                    | float     |
| Hilbert                 | Hilbert Transform attribute                                                                                                                | float     |
| envelope\_w\_phase        | Envelope weighted by phase                                                                                                                 | float     |
| envelope\_w\_freq         | Envelope weighted by frequency                                                                                                               | float     |
| Similarity              | Seismic similarity attribute                                                                                                                | float     |
| volume\_stati            | Volume statistics (could be mean, standard deviation, etc. - specify if known)                                                            | float     |
| Spectral\_Decomposition  | Output from spectral decomposition (likely multiple frequency bands)                                                                        | float     |
| Frequency\_Average       | Average frequency                                                                                                                       | float     |
| Energy                  | Seismic energy attribute                                                                                                                   | float     |
| Semblance               | Semblance attribute                                                                                                                         | float     |
| Texture                 | Seismic texture attribute                                                                                                                   | float     |
| AVO\_Porosity            | Porosity estimated from AVO analysis                                                                                                      | float     |
| AVO\_Lithology           | Lithology (rock type) estimated from AVO analysis                                                                                           | float     |
| Relief                  | Relief attribute (related to structural features)                                                                                            | float     |
| F3\_Seismic\_Data         | Raw seismic amplitude data                                                                                                                  | float     |
| F3\_Velocity\_Data        | Seismic velocity data                                                                                                                       | float     |

*Important Note:*  The provided code snippet only includes the *Volume\_of\_Shale* prediction.  The workflow for porosity prediction would be very similar, but you would train a separate model with "Porosity" as the target variable (ydata). You'd need to include a porosity column in your CSV.

**Libraries and Tools**
*   **Lasio:** For reading and writing LAS (Log ASCII Standard) well log files. *Important: While mentioned in the code, LAS files are not directly used in the provided snippet. This library would be essential if you are working with the original well log data.*
*   **Segyio:**  For reading and writing SEG-Y seismic data files. *Important: The code for generating the SEG-Y files for prediction in OpendTect is not included in the notebook. This is a critical step that needs to be added.*
*    **dask.array** :For working with large, multi-dimensional arrays that may not fit in memory
*   **dask.dataframe**: For working with large datasets that don't fit in memory, using a familiar Pandas-like API.
*   **mpl_toolkits.axes_grid1**:For creating grids of axes in Matplotlib, often used for adding colorbars or other supporting elements to plots.
*   **typing**:For specifying type hints in Python code, improving code readability and enabling static analysis tools
*   **OpendTect:** A seismic interpretation software package.  The models are trained in Python, but the final 3D prediction is performed in OpendTect. *Crucial:  The notebook doesn't show the OpendTect steps. You'll need to explain how to import the SEG-Y data and use the trained model within OpendTect.*

**Model Results**

| Model             | Training RMSE | Training R2 | Testing RMSE | Testing R2 |
| ----------------- | ------------- | ----------- | ------------ | ---------- |
| Linear Regression | 0.0077        | 0.36        | 0.0076       | 0.37       |
| Random Forest     | 0.0006        | 0.95        | 0.0007       | 0.94       |
| Neural Network    | 0.0007       | 0.94     | 0.0009       | 0.93       |

**Key Observations:**

*   The **Random Forest** and **Neural Network** models significantly outperform the **Linear Regression** model, indicating non-linear relationships between the seismic attributes and Vshale.
*   The **Random Forest** model shows some signs of **overfitting**, as the training R2 is much higher than the testing R2. This means the model is learning the training data too well and may not generalize as well to unseen data.
*   The **Neural Network** model provides a good balance of performance on both the training and testing sets, suggesting it's the most suitable model for this task, although there may be some mild overfitting.

