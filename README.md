# Large Scale Fraud Detection System

This project aims to build an end-to-end MLOps pipeline to detect fraud at scale, addressing Scalable Data Processing, Handling Imbalanced Data, Model Training & Experiment Tracking and Automated CI/CD Pipeline. 

---


## Table of Contents

1. [Project Overview]
2. [Features]
3. [Installation]
4. [Usage]
5. [Acknowledgments]

---

## Project Overview

### Tools

We use the below libraries and tools for the project
- Kaggle : To download the data
- ngrok : To connect and run MLFlow in the jupter notebook (first draft of project)
- optuna : To find the best parameters
- dvc : To manage our data versioning and also to create stages to run github workflow smoothly.
- yaml : To write the github workflow
- Xgboost - This is the selected model. 

### Screenshots or Demos

Below is our stages that will be executed. 

![Screenshot 2025-06-29 at 18 53 32](https://github.com/user-attachments/assets/606513eb-e822-497d-b5d5-49ed851168b2)
---

## Features

- This project is has a CI/CD pipeline which can be run with a pull request.
- This project using Mlflow tracking.
- This project uses DVC for tracking and for stages.
- This project uses a Xgboost model 

---

## Installation

### Prerequisites

1. Please refer to the requirements.txt for needed libraries.
2. You can clone the repository but please ensure you are setting up your kaggle api keys in the environment prior to proceeding. 

### Setup

Detailed steps to install and run the project:

```bash
bash
Copy code
# Clone the repository
git clone https://github.com/yourusername/yourprojectname.git](https://github.com/pullz6/Large-Scale-Fraud-Detection-System.git

# Navigate into the directory
cd Large-Scale-Fraud-Detection-System

# Install dependencies
pip install -r requirements.txt

```

---

## Usage

### Running the Application

- If you want to re-run this project you can create a pull request
- If you want to recreate and clone this project please follow the steps under Setup

### Configuration

Set you Kaggle api keys for authentication, you can follow these steps => https://medium.com/@wl8380/unlocking-kaggle-datasets-a-guide-to-obtaining-and-installing-your-api-key-65ca25a7ac7c

---

## Acknowledgments

DVC documentation - https://dvc.org/doc/command-reference/remove
