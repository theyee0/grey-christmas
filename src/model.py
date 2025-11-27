import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
import pandas as pd

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=None stride=1):
        super(Bottleneck, self).__init__()

        self.convolve0 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.convolve1 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.convolve2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels),
        )

        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        initial = x

        x = self.convolve0(x)
        x = self.convolve1(x)
        x = self.convolve2(x)

        if self.downsample:
            x = self.downsample(x)

        return self.relu(initial + x)


class Forecast(nn.Module):
    in_channels = 64

    def __init__(self, block_sizes, classes, channels=1):
        super(Forecast, self).__init__()

        self.convolve0 = nn.Conv1d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm0 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.max_pool = MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(block_sizes[0], channels=64)
        self.layer1 = self._make_layer(block_sizes[1], channels=128, stride=2)
        self.layer2 = self._make_layer(block_sizes[2], channels=256, stride=2)
        self.layer3 = self._make_layer(block_sizes[3], channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fully_connected = nn.Linear(512 * BottleNeck.expansion, classes)

    def _make_layer(self, blocks, channels, stride=1):
        target_channels = channels * BottleNeck.expansion

        downsampler = None

        if stride != 1 or self.in_channels != target_channels:
            downsampler = nn.Sequential(
                nn.Conv1d(self.in_channels, target_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(target_channels)
            )

        layers = [BottleNeck(self.in_channels, channels, downsample=downsampler, stride=stride)]

        for i in range(blocks - 1):
            layers.append(BottleNeck(self.in_channels, channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convolve0(x)
        x = self.batch_norm0(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fully_connected(x)

        return x


class WeatherDataset(Dataset):
    def __init__(self, weather_files):
        data = [pd.read_csv(weather_file,
                            usecols=["LOCAL_YEAR",
                                     "LOCAL_MONTH",
                                     "LOCAL_DAY",
                                     "MEAN_TEMPERATURE",
                                     "MIN_TEMPERATURE",
                                     "MAX_TEMPERATURE",
                                     "TOTAL_PRECIPITATION",
                                     "TOTAL_RAIN",
                                     "TOTAL_SNOW"])
                for weather_file in weather_files]

        years = data.groupby(pd.Grouper(key="LOCAL_YEAR"))

        for year in years:
            df.sort_values(by=["LOCAL_MONTH", "LOCAL_DAY"], ascending=[True, True], inplace=True)

        self.year_tensors = [self._christmas_tensor(year) if self._valid_year(year) for year in years]

    def _valid_year(self, year):
        if not ((year["LOCAL_MONTH"] == 1) & (year["LOCAL_DAY"] == 1)).any() or \
           not ((year["LOCAL_MONTH"] == 12) & (year["LOCAL_DAY"] == 31)).any()
            return False

        return True

    def _christmas_tensor(self, year):
        if not self._valid_year(year):
            return None

        X = self._date_to_tensor(year)

        filtered_day = year.query("LOCAL_MONTH == 12 and LOCAL_DAY == 25").loc(0)
        is_snowy = float(filtered_day["TOTAL_SNOW"]) > 0
        y = torch.Tensor([is_snowy, !is_snowy])

        return (X, y)

    def _date_to_tensor(self, year):
        gen_tensor = lambda x: torch.Tensor(
            x["MEAN_TEMPERATURE"],
            x["MIN_TEMPERATURE"],
            x["MAX_TEMPERATURE"],
            x["TOTAL_PRECIPITATION"],
            x["TOTAL_RAIN"],
            x["TOTAL_SNOW"],
        )

        year = year[year["LOCAL_MONTH"] != 2 or year["LOCAL_DAY"] != 29]

        return torch.Tensor([gen_tensor(day) for day in year.rows])

    def __len__(self):
        return len(self.year_tensors)

    def __getitem__(self, idx):
        return self.year_tensors[idx], self.success_tensors[idx]


class Trainer:
    def __init__(self, file_list):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model = None

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters())

        self.dataset = WeatherDataset(file_list)
        self.loader = Dataloader(dataset, batch_size=64)

        self.batches = len(loader)
        self.size = len(loader.dataset)

        self.type = torch.float

    def init(self):
        self.model = ...
        self.model.to(self.device)

    def train(self):
        self.model.train()

        size = len(loader)

        for batch, (X, y) in enumerate(self.loader):
            X, y = X.to(self.type).to(self.device), y.to(self.type).to(self.device)

            prediction = self.model(X)
            loss = self.criterion(prediction, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current, size = loss.item(), (batch + 1), self.batches
                print(f"Loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]\r")

        print()

    def test(self):
        self.model.eval()

        test_loss, correct = 0, 0

        with torch.no_grad():
            for (X, y) in dataloader:
                X, y = X.to(self.type).to(self.device), y.to(self.type).to(self.device)
                prediction = self.model(X)
                loss = self.criterion(prediction, y)

                test_loss += loss.item()
                correct += (prediction.argmax(1) == y).type(self.type).sum().item()
        test_loss /= self.batches
        correct /= self.size

        print(f"Test:\n   Accuracy: {100 * correct:>0.1f}%\n    Loss: {test_loss:>8f}\n")

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)


class Predictor:
    def __init__(self):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.type = torch.float

        self.model = None

    def load(self, filename):
        self.model = torch.load(filename, weights_only=True)
        self.model.to(device)

    def _date_to_tensor(self, year):
        gen_tensor = lambda x: torch.Tensor(
            x["MEAN_TEMPERATURE"],
            x["MIN_TEMPERATURE"],
            x["MAX_TEMPERATURE"],
            x["TOTAL_PRECIPITATION"],
            x["TOTAL_RAIN"],
            x["TOTAL_SNOW"],
        )

        year = year[year["LOCAL_MONTH"] != 2 or year["LOCAL_DAY"] != 29]

        return torch.Tensor([gen_tensor(day) for day in year.rows]).to(device)

    def predict(self, year):
        return self.predict_tensor(self._date_to_tensor(year)).item()

    def predict_tensor(self, year):
        return self.model(year)
