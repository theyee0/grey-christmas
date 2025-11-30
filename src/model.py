import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch import nn
from torch.optim import Adam

import pandas as pd

import numpy as np


class BottleNeck(nn.Module):
    """
    Implementation of Resnet bottleneck block, with 3 batch-normalized convolutional layers
    and residual
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, resample=None, stride=1):
        super(BottleNeck, self).__init__()

        # Inner layer to create the correct channels
        self.convolve0 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels, momentum=0.5),
            nn.ReLU()
        )

        # Middle layer to capture the properties of the tensor
        self.convolve1 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels, momentum=0.5),
            nn.ReLU()
        )

        # Outer layer to increase the number of channels
        self.convolve2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels * self.expansion, momentum=0.5),
        )

        self.stride = stride
        self.resample = resample
        self.relu = nn.ELU()

    def forward(self, x):
        initial = x.clone()

        x = self.convolve0(x)
        x = self.convolve1(x)
        x = self.convolve2(x)

        # Change the number of channels, if needed
        if self.resample is not None:
            initial = self.resample(initial)

        return self.relu(initial + x)


class Forecast(nn.Module):
    """
    Neural network module combining ResNet blocks to classify years as white christmasses or otherwise
    """
    in_channels = 64

    def __init__(self, block_sizes, classes, channels=1):
        super(Forecast, self).__init__()

        # Embedding layers
        self.convolve0 = nn.Conv1d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm0 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer0 = self._make_layer(block_sizes[0], channels=64)
        self.layer1 = self._make_layer(block_sizes[1], channels=128, stride=2)
        self.layer2 = self._make_layer(block_sizes[2], channels=256, stride=2)
        self.layer3 = self._make_layer(block_sizes[3], channels=512, stride=2)

        # Final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fully_connected = nn.Sequential(
            nn.Linear(512 * BottleNeck.expansion, 512),
            nn.ELU(),
            nn.Linear(512, 16),
            nn.ELU(),
            nn.Linear(16, classes),
        )

        self.flatten = nn.Flatten()

    def _make_layer(self, blocks, channels, stride=1):
        """
        Given the number of blocks and the input channels, create a ResNet layer
        """

        target_channels = channels * BottleNeck.expansion

        resample = None

        # Force resampling in case the output size is not correct
        if stride != 1 or self.in_channels != target_channels:
            resample = nn.Sequential(
                nn.Conv1d(self.in_channels, target_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(target_channels, momentum=0.5)
            )

        # Create and stack layers
        layers = [BottleNeck(self.in_channels, channels, resample=resample, stride=stride)]
        self.in_channels = target_channels
        for i in range(blocks - 1):
            layers.append(BottleNeck(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convolve0(x)
        x = self.batch_norm0(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.flatten(x)

        x = self.fully_connected(x)

        return x


class WeatherDataset(Dataset):
    """
    Given Canadian weather information, compile the temperature and precipitation into
    a pytorch Dataset
    """

    def __init__(self, weather_files):
        # Columns providing metadata for sorting
        self.metacols = [
            "CLIMATE_IDENTIFIER",
            "LOCAL_YEAR",
            "LOCAL_MONTH",
            "LOCAL_DAY",
        ]

        # Columns providing usable data for the model
        self.datacols = [
            "MEAN_TEMPERATURE",
            "MIN_TEMPERATURE",
            "MAX_TEMPERATURE",
            "TOTAL_PRECIPITATION",
            "TOTAL_RAIN",
            "TOTAL_SNOW",
        ]

        # All columns
        self.cols = self.metacols + self.datacols

        # Read and combine all files into a single pandas dataframe
        data = pd.concat([pd.read_csv(weather_file, usecols=self.cols)
                          for weather_file in weather_files])

        # Segment the pandas dataframe by station and year, then sort each year
        years = [x for _, x in data.groupby(by=["CLIMATE_IDENTIFIER", "LOCAL_YEAR"])]
        for year in years:
            year.sort_values(by=["LOCAL_MONTH", "LOCAL_DAY"], ascending=[True, True], inplace=True)

        # Convert valid years into pytorch tensors and store it into the data list
        self.year_tensors = [self._christmas_tensor(year) for year in years if self._valid_year(year)]

    def _valid_year(self, year):
        """
        Given a pandas dataframe representing a year, ensure that it contains all necessary data
        to be stored into the dataset
        """

        # Ensure that the dataset contains all days of the year by checking for first and last days
        firstday = ((year["LOCAL_MONTH"] == 1) & (year["LOCAL_DAY"] == 1)).any()
        lastday = ((year["LOCAL_MONTH"] == 12) & (year["LOCAL_DAY"] == 31)).any()
        if not (firstday or lastday):
            return False

        # Ensure the dataset has the correct number of days (otherwise not a full year)
        if len(year) < 365 or len(year) > 366:
            return False

        # Ensure that all columns contain valid data
        for col in self.cols:
            if year[col].isnull().any():
                return False

        return True

    def _date_to_tensor(self, year):
        """
        Given a pandas dataframe representing a year, convert it into a tensor
        """

        # Lambda function for turning rows into tensors
        gen_tensor = lambda x: torch.Tensor([x[col] for col in self.datacols])

        # Exclude leap years for simplicity
        year = year[(year["LOCAL_MONTH"] != 2) | (year["LOCAL_DAY"] != 29)]

        # Convert all rows into tensors and return the stack
        tensor_list = year.apply(gen_tensor, axis=1).values.tolist()
        return torch.stack(tensor_list, dim=1)

    def _christmas_tensor(self, year):
        """
        Given a pandas dataframe representing a year, make a final check that it represents
        a valid year, then convert it to a tensor
        """

        if not self._valid_year(year):
            return None

        # Convert the year into a tensor, avoiding the 12th month (so only data up to November needed
        X = self._date_to_tensor(year[year["LOCAL_MONTH"] != 12])

        # Create a tensor respresenting whether or not the year had snow on Christmas
        filtered_day = year.query("LOCAL_MONTH == 12 and LOCAL_DAY == 25").iloc[0]
        is_snowy = float(filtered_day["TOTAL_SNOW"]) > 0.0
        y = torch.Tensor([is_snowy, 1 - is_snowy])

        return (X, y)

    def __len__(self):
        """
        Return the number of items in the dataset
        """
        return len(self.year_tensors)

    def __getitem__(self, idx):
        """
        Return an item at a specified index in the dataset
        """
        return self.year_tensors[idx]


class Trainer:
    """
    Class for training a fresh model from scratch
    """

    def __init__(self, train_file_list, test_file_list):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

        self.type = torch.bfloat16

        # Initialize the model as an untrained forecaster
        self.model = Forecast([1, 3, 5, 1], classes=2, channels=6)
        self.model.to(self.device).to(self.type)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)

        # Create training Dataloader with weighted sampling to balance classes
        self.train_dataset = WeatherDataset(train_file_list)
        self.train_sampler = self._balanced_sampler(self.train_dataset)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=32, sampler=self.train_sampler
        )
        self.train_batches = len(self.train_loader)
        self.train_size = len(self.train_loader.dataset)

        # Create test Dataloader with weighted sampling to balance classes
        self.test_dataset = WeatherDataset(test_file_list)
        self.test_sampler = self._balanced_sampler(self.test_dataset)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1, sampler=self.test_sampler
        )
        self.test_batches = len(self.test_loader)
        self.test_size = len(self.test_loader.dataset)

    def _balanced_sampler(self, dataset):
        """
        Given a dataset, return a sampler to ensure that the classes (in this case snowy/not snowy)
        are approximately balanced to prevent underrepresentation in training
        """

        # Count the distinct labels in the dataset and assign them a weight
        labels = [y.argmax() for _, y in dataset]
        label_counts = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        label_weights = 1.0 / label_counts

        # Iterate over the labels and apply the determined weights to each
        sample_weights = torch.from_numpy(np.array([label_weights[l] for l in labels]))

        return WeightedRandomSampler(sample_weights, len(sample_weights))

    def train(self, epochs=10, batch_interval=1):
        """
        Perform the training procedure for a specified number of epochs and print status
        every `batch_interval` batches processed
        """

        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}:")

            # Set model into training mode
            self.model.train()

            # Iterate over batches from the dataloader
            for batch, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.type).to(self.device), y.to(self.type).to(self.device)

                prediction = self.model(X)
                assert prediction.size() == y.size(), "Expected prediction and known values to correspond"
                loss = self.criterion(prediction, y)

                # Optimize parameters
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Print status at regular intervals
                if batch % batch_interval == 0:
                    loss, current, size = loss.item(), (batch + 1), self.train_batches
                    print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]\r", end="")

            print()
            self.test()

    def test(self):
        """
        Test the model on the test set and print the results
        """

        self.model.eval()

        test_loss, correct = 0, 0

        with torch.no_grad():
            # Iterate over the examples from the testing set
            for i, (X, y) in enumerate(self.test_loader):
                X, y = X.to(self.type).to(self.device), y.to(self.type).to(self.device)
                prediction = self.model(X)
                loss = self.criterion(prediction, y)

                # Accumulate test loss and check count the fraction of correct responses
                test_loss += loss.item()
                correct += (prediction.argmax(1) == y.argmax(1)).sum().item()

        # Compute average test loss and fraction of correct predictions
        test_loss /= self.test_batches
        correct /= self.test_size

        print(f"Test:\n    Accuracy: {100 * correct:>0.1f}%\n    Loss: {test_loss:>8f}\n")

    def save(self, filename):
        """
        Export the model as a pytorch weights file
        """

        torch.save(self.model.state_dict(), filename)


class Predictor:
    """
    Class for wrapping the gritty details of the inference process
    """

    def __init__(self, filename):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.type = torch.bfloat16

        # Create the same model type as the training procedure
        self.model = Forecast([1, 3, 5, 1], classes=2, channels=6)
        self.model.to(self.device).to(self.type)

        # Load model weights
        self.model.load_state_dict(torch.load(filename, weights_only=True))

        self.datacols = [
            "MEAN_TEMPERATURE",
            "MIN_TEMPERATURE",
            "MAX_TEMPERATURE",
            "TOTAL_PRECIPITATION",
            "TOTAL_RAIN",
            "TOTAL_SNOW",
        ]

    def _date_to_tensor(self, year):
        """
        Given a pandas dataframe representing a year, convert it to a tensor
        """

        gen_tensor = lambda x: torch.Tensor([x[col] for col in self.datacols])

        # Exclude leap years
        year = year[(year["LOCAL_MONTH"] != 2) | (year["LOCAL_DAY"] != 29)]
        tensor_list = year.apply(gen_tensor, axis=1).values.tolist()

        return torch.stack(tensor_list, dim=1)

    def predict(self, year):
        """
        Given a pandas dataframe representing some year, predict the results as a tensor
        """
        return self.predict_tensor(self._date_to_tensor(year)).item()

    def predict_tensor(self, year):
        """
        Given a tensor representing some year, predict the rseults as a tensor
        """

        self.model.eval()

        year = year.unsqueeze(0)

        with torch.no_grad():
            return self.model(year.to(self.type).to(self.device))
