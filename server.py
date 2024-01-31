import argparse
from typing import List, Tuple

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import Metrics 

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="192.168.0.116:49999",
    help="gRPC server address (default '192.168.0.112:49999')",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=20,
    help="Number of rounds of federated learning (default: 20)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)

class Net(nn.Module):
    """Model (simple CNN adapted)."""

    def __init__(self):
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

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """This function averages the accuracy metric sent by the clients in an evaluate
    stage (i.e., clients received the global model and evaluate it on their local
    validation sets)."""
    # Multiply accuracy of each client by the number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_config(server_round: int):
    """Return a configuration with a static batch size and (local) epochs."""
    config = {
        "epochs": 5,  # Number of local epochs done by clients
        "batch_size": 32,  # Batch size to use by clients during fit()
    }
    return config

def main():
    args = parser.parse_args()
    print(args)

    model = Net()

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
      
    # Define strategy, Change the strategy as per your experiment
    strategy = fl.server.strategy.FedYogi(
        fraction_fit=args.sample_fraction,
        fraction_evaluate=args.sample_fraction,
        min_fit_clients=args.min_num_clients,
        min_available_clients=2,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        # proximal_mu=2.0, # this parameter should be used when using FedProx strategy. 
       )
        
    # Start Flower server

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    

if __name__ == "__main__":
    main()