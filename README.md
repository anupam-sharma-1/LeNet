# FashionMNIST Image Classification with PyTorch and Visualizations using TensorBoard

This code repository contains a Python script for image classification using the FashionMNIST dataset and PyTorch. FashionMNIST is a widely-used dataset for training and evaluating machine learning models for image classification tasks. The provided script covers the following key steps and features:

## Code Highlights

1. **Importing Libraries**: The code starts by importing essential libraries and configuring TensorFlow log levels.

2. **Data Preparation**:
   - It sets the device (CPU or GPU) based on the availability of CUDA.
   - Defines the batch size and image transformations, such as resizing and normalization.
   - Loads the FashionMNIST dataset for training and testing using PyTorch's torchvision.datasets.

3. **Data Visualization**:
   - The code includes a section to display a 3x3 grid of sample images from the dataset for visualization.

4. **Convolutional Neural Network (CNN)**:
   - A simple CNN model for image classification is defined. This model consists of convolutional and fully connected layers.

5. **Training the Model**:
   - The script trains the CNN model using the training data.
   - It tracks loss and accuracy metrics during training.

6. **Hyperparameter Tuning**:
   - The code supports hyperparameter tuning by allowing you to vary learning rates and batch sizes.
   - The model's performance is evaluated with different hyperparameter combinations.

7. **TensorBoard Integration**:
   - Training metrics are logged to TensorBoard for monitoring and analysis.

## Usage

1. Clone this repository to your local machine:

   ```
   git clone <repository_url>
   ```

2. Ensure you have the required dependencies installed.

3. Run the provided Python script to train and evaluate the CNN model. You can modify hyperparameters within the code.

## Dataset

The code utilizes the FashionMNIST dataset, a dataset of 28x28 grayscale images of clothing items. This dataset is included in the torchvision library for PyTorch.

## Contributions

Contributions to this project are encouraged. Feel free to fork the repository, make enhancements, and submit pull requests.

## License

This code is open-source and provided under a specific license. Refer to the [LICENSE](LICENSE) file for detailed licensing information.
