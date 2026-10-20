# this contains the dataloader for test and training
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import random



def load_training_data():
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )
    return training_data

def load_testing_data():
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    testing_data=datasets.FashionMNIST(
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

        self.targets = [int(label) for label in dataset.targets]

        self.indices_by_labels = {}
        # enumerate returns idx + value of that dataset
        for idx, label in enumerate(self.targets):
            self.indices_by_labels.setdefault(label,[]).append(idx)
        self.labels = sorted(set(self.targets))

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        image1, label1 = self.dataset[idx]

        label1 = int(label1)

        # during training 50/50 chance that we have equal chance of getting same or different pair

        random = random.Random()

        if random.Random() < 0.5:
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
        self.targets = [int(label) for label in dataset.targets]
        self.pairs = []

        self.indices_by_labels = {}

        for idx, label in enumerate(self.targets):
            self.indices_by_labels.setdefault(int(label),[]).append(idx)
        self.labels = sorted(set(self.targets))
        # print('keys',self.indices_by_labels.keys())
        rng = random.Random(seed)

        for _ in range(pair_per_type):
            # same pair
            label = rng.choice(self.labels)
            # print(type(label))
            # print(type(list(self.indices_by_labels.keys())[0]))
            
            idx1, idx2 = rng.sample(self.indices_by_labels[label],2)
            self.pairs.append((idx1,idx2, 1.0))

            # different pair
            label1, label2 = rng.sample(self.labels,2)

            idx1 = rng.choice(self.indices_by_labels[label1])
            idx2 = rng.choice(self.indices_by_labels[label2])

            self.pairs.append((idx1, idx2, 0.0))



    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, idx):

        idx1, idx2, same = self.pairs[idx]

        image1, _ = self.dataset[idx1]
        image2, _ = self.dataset[idx2]


        return (
            image1,
            image2,
            torch.tensor(same, dtype=torch.float32)
        )
def a():
    print(a)

if __name__ == '__main__':
    # we will check this function using main
    transform = v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True)])
    training_data = load_training_data()
    testing_data =load_testing_data()
    print('testing_data_shape', testing_data.data.shape)
    print('training_data_shape',training_data.data.shape)
    print('two check passed')

    # for creating the dataset for training and testing they require + and -ve pair
    training_data_siamese = SiameseTrainingData(training_data)
    print(len(training_data))
    testing_data = SiameseNetworkTestingData(testing_data,seed=0)
    print('len_of_testing data',len(testing_data))


    







    
