# WSL2.Windows11.TensorFlow
Installation guide and test file for validating TensorFlow in Ubuntu (WSL2) on Windows 11 with an NVIDIA GPU

# Project Overview:
To create a step-by-step guide to utilize an NVIDIA GPU installed on a Windows 11 laptop for Tensorflow workloads.  This requires using Ubuntu (Windows Subsystem for Linux 2) and installing and configuraing a Python TensorFlow environment in WSL, which enables passthrough to the GPU Driver installed on Windows. 

# Team Members:
Christopher Murphy

# Purpose:
While many cloud-based hardware acceleration solutions exist, leveraging a GPU to process TensorFlow workloads on a Windows 11 laptop can be challenging for a beginner.  There are specific dependencies between the NVIDIA GPU driver, CUDA versions, Python Versions and TensorFlow versions.  This guide hopes to simplify some of this confusion for NVIDIA GeForce RTX GPUs on Windows 11 environments.   

# Repository Information:
This repo contains:

*   A PDF guide for the installation and configuration of the WSL2 environment on Windows 11.
*   A powerpoint presentation summary of the PDF Guide.
*   A test file that can be run to validate a workload is being processed on the GPU.
