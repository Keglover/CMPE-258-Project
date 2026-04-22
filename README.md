# Deep Learning Adversarial Malware Detection Training Pipeline
Development repository for a modular AI/ML pipeline designed to experiment with model performance in the face of a GAN-like adversary.

Team Members: Kyle Glover

Group: 06


# Project Details

The goal of this project is to create a deep learning-based, GAN-style adversarial training pipeline, wherein the defender is pitted against an adversary that modifies training data in an attempt to subvert the defender, modifiying malicious lablled data to evade detection. As for data and inputs, raw PE binaries are used, and will be read in 2MB at a time - this is not quite a limitation, but rather a decision in order to remain comparable to [MalConv2](https://github.com/FutureComputing4AI/MalConv2), which will serve as a benchmark for the pipeline's output model, and will also evaluate the adversary's ability to evade detection.

Additionally, this pipeline is designed and intended to support full modularity, with users able to swap in different models and agents for both the adversary and defender based on what they specify in the configuration file. If time permits, the pipeline may be modified to allow for Reinforcement Learning style policy updates and feedback for the adversary and defender to work off of; however, this is a stretch goal, not a guarentee.

This project uses [EMBER2024](https://github.com/FutureComputing4AI/EMBER2024) for it's dataset, specifically the PE binary version of the dataset. The dataset for this project is NOT within the repository itself (in the interest of repository size), and must be downlaoded independently of this repository as of 13 April, 2026. I intend to have this done automatically when running the setup script later down the line.

Currently, plans for pipeline informatoin flow can be found in the Artifacts folder, along with any other non-code documents can be found. Additionally, the configuration file has been setup, and the beginnings of input digestion have been completed. Further development of the classes and scripts for defender and adversary creation from hyperparameter selection needs implementation, as well as testing and tuning suggested/default hyperparameters.


# Setup and Requirements

This repository provides a script (setup.py) in the base directory to automate downloading the EMBER2024 dataset, which is not present by default in this repository. **HOWEVER** It is highly recommended that the huggingface command line interface be downloaded and used before-hand in order to speed up the setup process - otherwise, the download will be massively rate limited.

## BEFORE you run setup:

Install the huggingface CLI, using the command ```curl -LsSf https://hf.co/cli/install.sh | bash``` on bash or terminal in MacOS/Linux.
Then, download the python module using ```pip install -U "huggingface_hub[cli]"```
Lastly, you will need a read-only token from your huggingface account to login thorugh the command-line. If you DO have an account or token, you can safely skip this step and go to the next pre-setup step. If you don't, go to [huggingface](https://huggingface.co/login) to create a huggingface account. After creating your account, go to your profile's settings page and go to access keys and create a read access key.
With your copied token, use the command ```hf auth login --token <YOUR_TOKEN>```. After this, you can freely run setup.py without being rate limited on downloading EMBER2024, significantly improving the download speed.
