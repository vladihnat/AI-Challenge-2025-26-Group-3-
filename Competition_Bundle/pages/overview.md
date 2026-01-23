# Overview of the Challenge
*** 
This challenge focuses on lightweight and efficient pollinator detection from image sequences using machine learning. Participants are invited to design models capable of identifying the presence of pollinators while respecting computational constraints.

The objective is not only to achieve good detection performance, but also to explore efficient and deployable solutions suitable for running 24/7 on limited hardware.

***


## Introduction
***
Pollinators play a crucial role in ecosystems and agriculture, yet monitoring their activity at scale remains challenging. Camera-based monitoring systems generate large volumes of image data that must often be processed continuously, making computational efficiency a key concern.

In this challenge, participants are provided with a dataset of images stored in .h5 files. Each .h5 file contains a sequence of images, similar to a stop-motion video, and is associated with a binary label:

 - 0 $\rightarrow$ No pollinator presence in the (frame / sequence ?)
 - 1 $\rightarrow$ Pollinator detected in the (frame / sequence ?)

The goal is to build a lightweight and accurate model capable of detecting pollinators from these image sequences. Solutions should be designed with the perspective of continuous deployment (24/7), where memory usage, inference time, and model complexity matter.

This competition encourages creative approaches, including but not limited to:
- Frame-based image classification
- Temporal or sequential modeling
- Models leveraging short-term historical information
- Efficient or compressed neural networks

## Competition Tasks
***
The main task of the competition is ** binary classification of image sequences**
* Input : 
	* `.h5` files containing a sequence of images 
* Output: 
	* One binary prediction per (sequence/frame?) 

Participants are free to choose how they exploit the data:

* Treating images independently
* Aggregating predictions over time
* Using temporal models (e.g. sequence models, temporal pooling, lightweight video models)

## Competition Phases
***
...

## How to join this competition?
***
TBD : Check the following instructions 

- Login or Create Account on [<ins>Codabench</ins>](https://www.codabench.org/) 
- Go to the `Starting Kit` tab
- Download the `Dummy Sample Submission`
- Go to the `My Submissions` tab
- Register in the Competition
- Submit the downloaded file


## Submissions
***
This competition allows **result-only submissions**.
Participants must submit a prediction file following the exact format described in the

⚠️ Important:
Although only predictions are submitted, participants are strongly encouraged to design models that are:

* Lightweight
* Efficient in inference
* Suitable for continuous (24/7) deployment

## Timeline
***


## Credits
***
This challenge is organized by [Université Paris-Saclay].

The dataset was collected and curated by [the INRAE team].
We thank all contributors involved in data collection, annotation, and platform support.

## Contact
***
For questions or issues related to the competition, please contact: ...
