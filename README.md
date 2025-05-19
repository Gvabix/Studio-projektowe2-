# Fake News Classification using Graph Neural Networks

## Description

This project compares the performance of the **HPNF (Hierarchical Propagation Neural Network)** model with the **RST** model, as described in [this paper](https://arxiv.org/abs/1903.09196). The goal is to classify **fake** vs **real** news articles using **graph-based methods**. The dataset consists of both **graph data** (from [Fake News Propagation](https://github.com/mdepak/fake-news-propagation)) and **textual data** (from [Fakenews-dataset](https://github.com/mbzuai-nlp/Fakenews-dataset)).

- **HPNF** is a hierarchical graph neural network that propagates information in a hierarchical manner for more efficient graph-based classification tasks.
- **DPLP** (available [here](https://github.com/jiyfeng/DPLP)) is a reference model for comparison, which analyzes rhetorical structures (RST) in news articles and uses these structures for fake news classification.

## Data

- **Graph data**: This includes relationships between tweets (nodes) and replies (edges). The `prepare_graph_data.py` script processes this raw data and converts it into graph format for use in the model.
- **Text data**: The textual content of news articles is used for classification and analysis.

## How it works

1. **Graph Data Processing**: The `prepare_graph_data.py` script processes raw JSON data, creating graphs where nodes represent tweets and edges represent replies. These graphs are then used as input for the HPNF model.
2. **Model Training**: The `train_hpnf.py` script trains the **HPNF** model on the processed graph data, performing the task of fake news classification.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Gvabix/Studio-projektowe2-.git
   ```

2. **Prepare the Graph Data**:

   Before training, you need to preprocess the graph data. The dataset for graphs is stored as a **ZIP file** in the repository, so you will need to unzip it first.

   - Unzip the `graphs.zip` file into a folder named `graphs/` in the data directory.

3. **Optionally, create a virtual environment**:

   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux =>  preferably create conda environment
   ```

4. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Graph Data Preparation**:

   Before training the model, you need to process the graph data using the `prepare_graph_data.py` script:

   ```bash
   python datasets/prepare_graph_data.py
   ```

   This will preprocess the raw graph data into the format required for the model.

6. **Train the Model**:

   After preparing the data, you can train the model by running:

   ```bash
   python models/hpnf/train_hpnf.py
   ```

This will:

- Load and preprocess the dataset.
- Train the **HPNF** model for fake news classification.
- Save the trained model to `models/hpnf/checkpoints/hpnf.pt`.

7. **Evaluate the model**
   After running the model, you can evaluate the results by running:

```bash
python utils/evaluate_hpnf.py
```
> All results will be saved to `results/hpnf/`
   
---

## References/Citation

@article{shu2019hierarchical, title={Hierarchical propagation networks for fake news detection: Investigation and exploitation}, author={Shu, Kai and Mahudeswaran, Deepak and Wang, Suhang and Liu, Huan}, journal={arXiv preprint arXiv:1903.09196}, year={2019} }

@article{shu2018fakenewsnet, title={FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media}, author={Shu, Kai and Mahudeswaran, Deepak and Wang, Suhang and Lee, Dongwon and Liu, Huan}, journal={arXiv preprint arXiv:1809.01286}, year={2018} }
