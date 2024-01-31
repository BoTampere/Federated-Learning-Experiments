import argparse
import copy
import random
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="192.168.0.116:49999",    # replace with your server IP address
    help=f"gRPC server address (deafault '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)
parser.add_argument(
    "--iid",
    action="store_true",
    help="If you use iid use true else false",
)


warnings.filterwarnings("ignore", category=UserWarning)

NUM_CLIENTS = 10  # Number of clients = 10 for IID and 6 for non-IID
NUM_CLASSES = 10  # Number of classes in CIFAR-10


class Net(nn.Module):
    """Model (simple CNN adapted)."""
    def __init__(self): # baseline 2
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x


def train(net, trainloader, valloader, optimizer, epochs, device, global_params, config):
    """Train the model on the training set."""
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    correct, loss = 0, 0.0
    running_loss = 0.0
    total = 0
    total_val = 0
    val_correct = 0
    proximal_term = 0.0

    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            outputs = net(images.to(device))
            # In case where FedPRox strategy is used change the loss, uncomment the below commented part.

            # for local_weights, global_weights in zip(net.parameters(), global_params):
            #     proximal_term += (local_weights - global_weights).norm(2)
            # loss = criterion(outputs, labels.to(device)) + (config["proximal_mu"] / 2) * proximal_term
            # loss.backward(retain_graph=True)

            # Comment the below part related to loss when using FedProx.
            loss = criterion(outputs, labels.to(device))
            loss.backward()  

            optimizer.step()
            running_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%")

        with torch.no_grad():
            for images, labels in tqdm(valloader):
                outputs = net(images.to(device))
                labels = labels.to(device)

                total_val += labels.size(0)
                val_correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            epoch_val_accuracy = 100 * val_correct / total_val

            val_accuracies.append(epoch_val_accuracy)
            print(f"Epoch {epoch + 1}/{epochs}: Val Accuracy: {epoch_val_accuracy:.2f}%")
    
    return train_accuracies, val_accuracies


def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    test_acc = 100*accuracy
    print(f"Test Accuracy: {test_acc:.2f}%")
    return loss, accuracy


def prepare_dataset(iid=True):
    """Get CIFAR-10 and return client partitions and global testset."""
    dataset = CIFAR10
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trf = Compose([ToTensor(), norm])
    trainset = dataset("./data", train=True, download=True, transform=trf)
    testset = dataset("./data", train=False, download=True, transform=trf)

    print("Partitioning dataset (IID)..." if iid else "Partitioning dataset (non-IID)...")

    if iid:
        # Split trainset into `num_partitions` trainsets (IID)
        num_images = len(trainset) // NUM_CLIENTS
        partition_len = [num_images] * NUM_CLIENTS

        trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))
    else:
        # Define the non-IID partitioning strategy
        NUM_SAMPLES_PER_CLASS = 1000  # samples per class

        trainsets = []
        for _ in range(NUM_CLIENTS):
            # Sample non-IID data for each client
            selected_classes = random.sample(range(NUM_CLASSES), 5)  # Shuffle classes

            # selected_classes = random.sample(range(NUM_CLASSES), num_selected_classes) pass num_selected_classes as argument to function

            client_partition = []
            for class_id in selected_classes:
                # Select a subset of samples from each class
                class_indices = [i for i, (_, label) in enumerate(trainset) if label == class_id]

                # Randomly select a subset of samples from each class
                selected_indices = random.sample(class_indices, NUM_SAMPLES_PER_CLASS)

                # Add the selected samples to the client's partition
                client_partition.extend([trainset[i] for i in selected_indices])

            trainsets.append(client_partition)

    val_ratio = 0.1
    train_partitions = []
    val_partitions = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)  # 10% for validation
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        train_partitions.append(for_train)
        val_partitions.append(for_val)

    return train_partitions, val_partitions, testset


# Flower client, adapted from Pytorch quickstart/simulation example
class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that trains a CNN model for CIFAR-10 """

    def __init__(self, model, trainset, valset, testset):
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.model = model
        
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        # Read hyperparameters from config set by the server
        batch, epochs = config["batch_size"], config["epochs"]
        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        valloader = DataLoader(self.valset, batch_size=32)
        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        global_params = copy.deepcopy(self.model).parameters()
        # Train
        train(self.model, trainloader, valloader, optimizer, epochs=epochs, device=self.device, global_params=global_params,
                config=config)
        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        # Construct dataloader
        testloader = DataLoader(self.testset, batch_size=32)
        # Evaluate
        loss, accuracy = test(self.model, testloader, device=self.device)
        # Return statistics
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}


def main():
    args = parser.parse_args()
    print(args)
    use_iid = args.iid
    assert args.cid < NUM_CLIENTS

    # Instantiate model
    model = Net()

    # Download CIFAR-10 dataset and partition it
    trainsets, valsets, testsets = prepare_dataset(use_iid)

    # Start Flower client setting its associated data partition
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FlowerClient(
            model, trainset=trainsets[args.cid], valset=valsets[args.cid], testset = testsets
        ),
    )


if __name__ == "__main__":
    main()