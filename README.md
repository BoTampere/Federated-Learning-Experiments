# Federated Learning on Embedded Devices(Raspberry Pi cluster) with Flower Framework

## Overview

This repository contains code for implementing federated learning on embedded devices using Flower, a federated learning framework. The setup involves 6 Raspberry Pis acting as clients and a Linux VM serving as the central server running on a laptop connected via WIFI TCP/IP.

## Requirements

- Raspberry Pis (At least 2 - any number of clients as per your requirement.)
- Ubuntu Linux VM (for the server)
- Python 3. x
- Flower framework

## Hardware Setup

### 1. Server
Any device or laptop can be configured to be a server. Still, I used the Oracle VM Virtual box to create an Ubuntu Virtual Machine as Flower offers better compatibility and support.

Follow this link to set up an Ubuntu VM: https://ubuntu.com/tutorials/how-to-run-ubuntu-desktop-on-a-virtual-machine-using-virtualbox#1-overview

After creating the VM, open or free a port on which the server can be hosted. In my case, I chose 49999.

### 2. Client

Set up the Raspberry Pis with the Ubuntu Server of your preferred version using Raspberry Pi Imager.

Follow this tutorial link for setup: https://ubuntu.com/tutorials/how-to-install-ubuntu-on-your-raspberry-pi#1-overview

I used the Ubuntu Server 22.04.3 LTS(64bit) version for all my clients.

Just so you know, avoid using Ubuntu Desktop on Raspberry Pi as it occupies more storage and might lead to memory issues for Federated Learning.

## Setup Python Virtual Environment and install dependencies.

1. Set up a Python Virtual environment in the server and the clients.

Follow this tutorial to set up a virtual environment for your project: https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/

2. Clone the repository in the project folder where you have the Virtual Environment for both server and Raspberry Pi devices.

  
```bash
git clone https://github.com/BoTampere/Federated-Learning-Experiments.git

cd federated-learning-embedded
 ```

3. Install required Python packages in the server and clients:

  
```bash
pip install -r requirements.txt
```  
Note: Install pip and other general software necessary on the Raspberry Pi clients if not pre-installed with Ubuntu OS.

## Usage

### Server

Run the server on the Ubuntu VM:

```bash
python server.py
```
### Clients

Run the client code on each Raspberry Pi:

#### Standard Client

```bash
python client.py
```
#### Differential Privacy Client

```bash
python differentialprivacy_client.py
```
## Configuration of Hyperparameters

The below hyperparameters are consistent across the experiments and these can be modified as required.

1. Optimizer: Adam, 
2. Loss Function: Cross Entropy Loss, 
3. Learning Rate: 0.001, 
4. Epochs: 5, 
5. Number of Rounds: 20.

## Acknowledgments

Special thanks to the Flower framework contributors.

## Disclaimer

This project is for educational purposes. Be aware of potential security concerns when deploying federated learning in real-world scenarios.

Happy Federated Learning!
