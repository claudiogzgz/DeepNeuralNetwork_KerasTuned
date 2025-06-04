# DeepNeuralNetwork_KerasTuned

Optimized Deep Neural Network for regression using Keras Tuner, with model comparison and performance benchmarking.

## Project Overview

This project focuses on building a high-performance deep neural network (DNN) for a regression task, leveraging the Keras Tuner library for automated hyperparameter optimization. The final model is benchmarked against traditional machine learning algorithms to evaluate its effectiveness.

## Key Features

-  Deep Neural Network with ReLU activations and He initialization  
-  Hyperparameter tuning using **Keras Tuner (RandomSearch)**  
-  L2 regularization for generalization  
-  Early stopping to prevent overfitting  
-  Comparison against:
  - Linear Regression  
  - Random Forest Regressor  
-  Evaluation using **MSE**, **MAE**, and **RMSE** on validation set  
-  Predictions generated for a provided test set  

## Results

| Model              | MSE        | MAE       | RMSE     |
|-------------------|------------|-----------|----------|
| Linear Regression | 1173.126   | 26.690    | 34.251   |
| Random Forest     | 628.668    | 15.131    | 25.073   |
| Untuned DNN       | 3180.912   | 41.431    | 56.400   |
| **Tuned DNN**     | **56.269** | **5.682** | **7.501** |

 TThe tuned DNN significantly outperformed all other models across MSE, MAE, and RMSE — achieving over 70% lower RMSE than the next best model, Random Forest.

 The tuned DNN outperformed all other models across all metrics.

## Technologies

- Python 3
- TensorFlow / Keras
- Keras Tuner
- Scikit-learn
- Pandas, NumPy, Matplotlib

## Files

- `TunedDNN_Regression_Benchmark.ipynb`: full modeling, tuning, and benchmarking workflow
- `README.md`: project overview

## Contact

Created by [Claudio Gonzalez](https://github.com/claudiogzgz)  
Feel free to connect or reach out via GitHub if you have questions or ideas!
