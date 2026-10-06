import torch.nn as nn

class Solubility_Predictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()


        self.model = nn.Sequential(

            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),


            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),


            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),


            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            # Output target (LogS)
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)