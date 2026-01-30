# Overview of the Challenge
*** 
This challenge focuses on lightweight and efficient pollinator detection from image sequences using machine learning. Participants are invited to design models capable of identifying the presence of pollinators while respecting computational constraints.

The objective is not only to achieve good detection performance, but also to explore efficient and deployable solutions suitable for running 24/7 on limited hardware.

***


## Introduction
***
Pollinators play a crucial role in ecosystems and agriculture, yet monitoring their activity at scale remains challenging. Camera-based monitoring systems generate large volumes of image data that must often be processed continuously, making computational efficiency a key concern.

In this challenge, participants are provided with a dataset of images stored in a .h5 files. This file contains a sequence of images, similar to a stop-motion video, and is associated with a binary label:

 - 0 $\rightarrow$ No pollinator presence in the frame
 - 1 $\rightarrow$ Pollinator presence detected in the frame

The goal is to build a lightweight and accurate model capable of detecting pollinators from these image sequences. The solution should be designed with the perspective of a continuous deployment (24/7), where memory usage, inference time, and model complexity matter.

This competition encourages approaches, such as :
- Frame-based image classification
- Temporal or sequential modeling
- Models leveraging short-term historical information
- Efficient or compressed neural networks

But other ideas could also be explored 

## Competition Tasks
***
This competition features a **single task: frame-level binary classification on image sequences**.

**Task Definition**:
Participants must submit a model that processes .h5 files containing sequences of image frames and produces a binary prediction for each frame indicating pollinator presence.

Input:
A .h5 file containing an ordered sequence of image frames.

Output:
A binary prediction for each frame in the sequence:

## Competition Phases
***
The competition is organized into two evaluation phases: a "public" phase and a "private" phase.

**Phase 1 — "Public" Phase** :

During the public phase, participants submit their models to the Codabench platform,submissions are evaluated automatically upon upload.
The results of the submission are displayed on a public leaderboard, allowing participants to track and compare performance throughout the competition.

**Phase 2 — "Private" Phase**:

At the end of the competition, all submitted models are evaluated on a separate, unseen private test set.
The private test set is disjoint from the "public" evaluation data and is used to assess the generalization performance of submitted models.
Evaluation during this phase is performed only once, after the submission deadline.
The resulting private leaderboard is hidden from participants and is used to determine the final rankings.

This two-phase evaluation protocol ensures fair comparison between participants and mitigates overfitting to the public leaderboard.

## How to join this competition?
***

- Login or Create Account on [<ins>Codabench</ins>](https://www.codabench.org/) 
- Go to the `Starting Kit` tab
- Download the `Dummy Sample Submission`
- Go to the `My Submissions` tab
- Register in the Competition
- Submit the downloaded file


## Submissions
***
Submissions consist of a trained model compatible with the Codabench execution environment.

For each submission, Codabench automatically evaluates the model on a private test set of unseen .h5 files. 

The model is expected to:
- Load the input .h5 file
- Process the full sequence of frames
- Output one prediction per frame in the required format

All performance metrics are computed on the private test set to ensure fair and reproducible evaluation.

## Credits
***
This challenge is organized by **Université Paris-Saclay**.

The dataset was collected and curated by the **INRAE team**.
We thank all contributors involved in data collection, annotation, and platform support.

**Team Members**:

- Zeying Li
- Zuolong Charlotte
- Pras Baptiste
- Leiva Martin
- Peña Castaño Javier
- Herrera Nativi Vladimir

**Teaching Team** : 

- Ihsan Ullah
- Hieu Khuong T. G.
- Lisheng Sun

## Contact
***
For questions or issues related to the competition, please contact: baptiste.pras@universite-paris-saclay.fr
