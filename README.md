# CMPE-258-Project
Development repository for a modular AI/ML pipeline designed to experiment with model performance in the face of a GAN-like adversary.

Team Members: Kyle Glover
Group: 06

=========================================================
# Project Details

The goal of this project is to create a deep learning-based, GAN-style adversarial training pipeline, wherein the defender is pitted against an adversary that modifies training data in an attempt to subvert the defender, modifiying malicious lablled data to evade detection. As for data and inputs, raw PE binaries are used, and will be read in 2MB at a time - this is not quite a limitation, but rather a decision in order to remain comparable to [MalConv2](https://github.com/FutureComputing4AI/MalConv2), which will serve as a benchmark for the pipeline's output model, and will also evaluate the adversary's ability to evade detection.

Additionally, this pipeline is designed and intended to support full modularity, with users able to swap in different models and agents for both the adversary and defender. If time permits, the pipeline may be modified to allow for Reinforcement Learning style policy updates and feedback for the adversary and defender to work off of; however, this is a stretch goal, not a guarentee.

This project uses [EMBER2024](https://github.com/FutureComputing4AI/EMBER2024) for it's dataset, specifically the PE binaries. The dataset for this project is NOT within the repository itself, and must be downlaoded independently of this repository as of 13 April, 2026. I intend to have this done automatically when running the setup script later down the line.

Currently, plans for pipeline informatoin flow can be found in the Artifacts folder, along with any other non-code documents can be found. Additionally, the configuration file has been setup, and the beginnings of input digestion have been completed. Further development of the classes and scripts for defender and adversary creation from hyperparameter selection needs implementation, as well as testing and tuning suggested/default hyperparameters.

=========================================================
# Setup and Requirements

A setup script will be necessary to make sure that every dependency is handled correctly and the code is able to be run independent of the host architecture. THIS IS NOT CURRENTLY IMPLEMENTED
