# this contains the dataloader for test and training

from torch.utils.data import Dataset, Dataloader
from torchvision import dataset 
from torchvision.transforms import v2
import random



def load_training_data(transform):
    transfrom = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data = Dataset.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )
    return training_data

def load_testing_data(transform):
    transfrom = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    testing_data=Dataset.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=transform

    )

    return testing_data


# class for training pairs and testing pairs

class SiameseTrainingData(Dataset):

    def __init__(self,dataset):

        self.dataset = dataset

        self.target = [int(label) for label in dataset.targets]

        self.indices_by_labels = {}
        # enumerate returns idx + value of that dataset
        for idx, label in enumerate(dataset.target):
            self.indices_by_labels.setdefault(label,[]).append(idx)
        self.labels = sorted(set(self.target))

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        image1, label1 = self.dataset[idx]

        label1 = int(label1)

        # during training 50/50 chance that we have equal chance of getting same or different pair

        random = random.Random()

        if random < 0.5:
            # generate same pair

            idx2 = random.choice(self.indices_by_labels[label1]) 

            same = 1.0

        else:

            label2 = random.choice([label for label in self.labels if label != label1])

            idx2 = random.choice(self.indices_by_labels[label2])

            same = 0.0

        image2, _ = self.dataset[idx2]


        return (
            image1,
            image2,
            torch.tensor(same, dtype=torch.float32)
        )


# the testing data need to same as we need our model evaluation to have same comparision

class SiameseNetworkTestingData(Dataset):

    def __init__(self,dataset,seed,pair_per_type=1000):
        
        self.dataset = dataset
        self.seed = seed
        self.targets = [int(label) for label in dataset.target]
        self.pairs = []

        self.indices_by_labels = {}

        for idx, label in enumerate(dataset.target):
            self.indices_by_labels.setdefault(label,[]).append(idx)
        self.labels = sorted(set(self.targets))

        rng = random.Random(seed)

        for _ in range(pairs_per_type):

            label = rng.choice(self.labels)
            idx1, idx2 = rng.choice(self.indices_by_labels[label],2)
            self.pairs.append((idx1,idx2, 1.0))


            label1, label2 = rng.sample(labels,2)

            idx1 = rng.choice(self.indices_by_labels[label1])
            idx2 = rng.choice(self.indices_by_labels[label2])

            self.pairs.append((idx1, idx2, 0.0))



    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        idx1, idx2, same = self.pairs[idx]

        image1, _ = self.dataset[idx1]
        image2, _ = self.dataset[idx2]


        return (
            image1,
            image2,
            torch.tensor(same, dtype=torch.float32)
        )






    
