Cancer Prediction from DNA Methylation Site Identification

DNA Sequence Cancer Diagnosis is a machine learning project that aims to predict cancer based on DNA methylation site identification. The project focuses on leveraging multidimensional information within gene sequences to improve prediction accuracy compared to traditional methods like Whole-genome bisulfite sequencing (WGBS).
Overview

DNA methylation plays a crucial role in gene expression regulation by influencing DNA stability and chromosome structure. This project addresses the limitations of traditional methods and employs machine learning to predict DNA methylation modification sites efficiently.
Features

Predicts DNA methylation modification sites using machine learning.
Three different encoding methods for gene sequences: Binary encoding of Position Feature (BPF), Coding of nucleic acid chemical properties (NCP), and Coding of Dinucleotide physical and chemical properties (DPCP).
Utilizes a Convolutional Neural Network (CNN) named MEDCNN for multidimensional feature encoding.

Getting Started
Prerequisites

    Python
    Libraries: NumPy, Pandas, Scikit-learn, TensorFlow, Keras

Installation

bash

    git clone https://github.com/devatraj/DNA-Sequence-Cancer-Diagnosis.git
    cd DNA-Sequence-Cancer-Diagnosis
    pip install -r requirements.txt

Usage

Prepare your dataset with entries containing index, label, and gene sequences.
Run the preprocessing script:

bash

    python preprocess.py --input_data dataset.csv --output_features features.npy --output_labels labels.npy

Train the model:

bash

    python train.py --features features.npy --labels labels.npy --epochs 10

Make predictions:

bash

    python predict.py --input_sequence new_sequence.txt

Data

The dataset includes entries with an index, label (0 for not methylated, 1 for methylated), and gene sequences.

Example:

    Index | Label | Text
    0     | 1     | TGTGGAGGGAGAGGCGCGGGCGGGAACTGCTGCTGTGCATG

Model Training

The model is trained using a Convolutional Neural Network (CNN) named MEDCNN. Three encoding methods (BPF, NCP, DPCP) are used to represent gene sequences as multidimensional feature matrices.

Evaluation

The model's performance is evaluated based on metrics such as accuracy, precision, recall, and F1 score.

Results

Provide insights into the model's predictions and any patterns discovered.

Contributing

Feel free to contribute by submitting issues, proposing new features, or enhancing the existing ones.

License

This project is licensed under the Apache license.

Acknowledgements

Thank you to the research community for advancements in computational methods for DNA methylation prediction.
