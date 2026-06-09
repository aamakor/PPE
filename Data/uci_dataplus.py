# UCI Adult dataset, Data loading and preprocessing for 5 classification tasks: 1) age >= 40, 2) education level >= Bachelors, 3) marital status = Never married, 4) race = White, 5) sex = Male.
# For more details on the dataset and features, see https://archive.ics.uci.edu/ml/datasets/Adult

from cProfile import label
from pathlib import Path
import torch
from collections import OrderedDict
import numpy as np
import gzip
import os

uci_info = '''
age: label.
class of worker: Not in universe, Federal government, Local government, Never worked, Private, Self-employed-incorporated, Self-employed-not incorporated, State government, Without pay.
detailed industry recode: 0, 40, 44, 2, 43, 47, 48, 1, 11, 19, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 4, 42, 45, 5, 15, 16, 22, 29, 31, 50, 14, 17, 18, 28, 3, 30, 41, 46, 51, 12, 13, 21, 23, 26, 6, 7, 9, 49, 27, 8, 10, 20.
detailed occupation recode: 0, 12, 31, 44, 19, 32, 10, 23, 26, 28, 29, 42, 40, 34, 14, 36, 38, 2, 20, 25, 37, 41, 27, 24, 30, 43, 33, 16, 45, 17, 35, 22, 18, 39, 3, 15, 13, 46, 8, 21, 9, 4, 6, 5, 1, 11, 7.
education: label.
wage per hour: continuous.
enroll in edu inst last wk: Not in universe, High school, College or university.
marital stat: label.
major industry code: Not in universe or children, Entertainment, Social services, Agriculture, Education, Public administration, Manufacturing-durable goods, Manufacturing-nondurable goods, Wholesale trade, Retail trade, Finance insurance and real estate, Private household services, Business and repair services, Personal services except private HH, Construction, Medical except hospital, Other professional services, Transportation, Utilities and sanitary services, Mining, Communications, Hospital services, Forestry and fisheries, Armed Forces.
major occupation code: Not in universe, Professional specialty, Other service, Farming forestry and fishing, Sales, Adm support including clerical, Protective services, Handlers equip cleaners etc , Precision production craft & repair, Technicians and related support, Machine operators assmblrs & inspctrs, Transportation and material moving, Executive admin and managerial, Private household services, Armed Forces.
race: label.
hispanic origin: Mexican (Mexicano), Mexican-American, Puerto Rican, Central or South American, All other, Other Spanish, Chicano, Cuban, Do not know, NA.
sex: label.
member of a labor union: Not in universe, No, Yes.
reason for unemployment: Not in universe, Re-entrant, Job loser - on layoff, New entrant, Job leaver, Other job loser.
full or part time employment stat: Children or Armed Forces, Full-time schedules, Unemployed part- time, Not in labor force, Unemployed full-time, PT for non-econ reasons usually FT, PT for econ reasons usually PT, PT for econ reasons usually FT.
capital gains: continuous.
capital losses: continuous.
dividends from stocks: continuous.
tax filer stat: Nonfiler, Joint one under 65 & one 65+, Joint both under 65, Single, Head of household, Joint both 65+.
region of previous residence: Not in universe, South, Northeast, West, Midwest, Abroad.
state of previous residence: Not in universe, Utah, Michigan, North Carolina, North Dakota, Virginia, Vermont, Wyoming, West Virginia, Pennsylvania, Abroad, Oregon, California, Iowa, Florida, Arkansas, Texas, South Carolina, Arizona, Indiana, Tennessee, Maine, Alaska, Ohio, Montana, Nebraska, Mississippi, District of Columbia, Minnesota, Illinois, Kentucky, Delaware, Colorado, Maryland, Wisconsin, New Hampshire, Nevada, New York, Georgia, Oklahoma, New Mexico, South Dakota, Missouri, Kansas, Connecticut, Louisiana, Alabama, Massachusetts, Idaho, New Jersey.
detailed household and family stat: Child <18 never marr not in subfamily, Other Rel <18 never marr child of subfamily RP, Other Rel <18 never marr not in subfamily, Grandchild <18 never marr child of subfamily RP, Grandchild <18 never marr not in subfamily, Secondary individual, In group quarters, Child under 18 of RP of unrel subfamily, RP of unrelated subfamily, Spouse of householder, Householder, Other Rel <18 never married RP of subfamily, Grandchild <18 never marr RP of subfamily, Child <18 never marr RP of subfamily, Child <18 ever marr not in subfamily, Other Rel <18 ever marr RP of subfamily, Child <18 ever marr RP of subfamily, Nonfamily householder, Child <18 spouse of subfamily RP, Other Rel <18 spouse of subfamily RP, Other Rel <18 ever marr not in subfamily, Grandchild <18 ever marr not in subfamily, Child 18+ never marr Not in a subfamily, Grandchild 18+ never marr not in subfamily, Child 18+ ever marr RP of subfamily, Other Rel 18+ never marr not in subfamily, Child 18+ never marr RP of subfamily, Other Rel 18+ ever marr RP of subfamily, Other Rel 18+ never marr RP of subfamily, Other Rel 18+ spouse of subfamily RP, Other Rel 18+ ever marr not in subfamily, Child 18+ ever marr Not in a subfamily, Grandchild 18+ ever marr not in subfamily, Child 18+ spouse of subfamily RP, Spouse of RP of unrelated subfamily, Grandchild 18+ ever marr RP of subfamily, Grandchild 18+ never marr RP of subfamily, Grandchild 18+ spouse of subfamily RP.
detailed household summary in household: Child under 18 never married, Other relative of householder, Nonrelative of householder, Spouse of householder, Householder, Child under 18 ever married, Group Quarters- Secondary individual, Child 18 or older.
instance weight: ignore.
migration code-change in msa: Not in universe, Nonmover, MSA to MSA, NonMSA to nonMSA, MSA to nonMSA, NonMSA to MSA, Abroad to MSA, Not identifiable, Abroad to nonMSA.
migration code-change in reg: Not in universe, Nonmover, Same county, Different county same state, Different state same division, Abroad, Different region, Different division same region.
migration code-move within reg: Not in universe, Nonmover, Same county, Different county same state, Different state in West, Abroad, Different state in Midwest, Different state in South, Different state in Northeast.
live in this house 1 year ago: Not in universe under 1 year old, Yes, No.
migration prev res in sunbelt: Not in universe, Yes, No.
num persons worked for employer: continuous.
family members under 18: Both parents present, Neither parent present, Mother only present, Father only present, Not in universe.
country of birth father: Mexico, United-States, Puerto-Rico, Dominican-Republic, Jamaica, Cuba, Portugal, Nicaragua, Peru, Ecuador, Guatemala, Philippines, Canada, Columbia, El-Salvador, Japan, England, Trinadad&Tobago, Honduras, Germany, Taiwan, Outlying-U S (Guam USVI etc), India, Vietnam, China, Hong Kong, Cambodia, France, Laos, Haiti, South Korea, Iran, Greece, Italy, Poland, Thailand, Yugoslavia, Holand-Netherlands, Ireland, Scotland, Hungary, Panama.
country of birth mother: India, Mexico, United-States, Puerto-Rico, Dominican-Republic, England, Honduras, Peru, Guatemala, Columbia, El-Salvador, Philippines, France, Ecuador, Nicaragua, Cuba, Outlying-U S (Guam USVI etc), Jamaica, South Korea, China, Germany, Yugoslavia, Canada, Vietnam, Japan, Cambodia, Ireland, Laos, Haiti, Portugal, Taiwan, Holand-Netherlands, Greece, Italy, Poland, Thailand, Trinadad&Tobago, Hungary, Panama, Hong Kong, Scotland, Iran.
country of birth self: United-States, Mexico, Puerto-Rico, Peru, Canada, South Korea, India, Japan, Haiti, El-Salvador, Dominican-Republic, Portugal, Columbia, England, Thailand, Cuba, Laos, Panama, China, Germany, Vietnam, Italy, Honduras, Outlying-U S (Guam USVI etc), Hungary, Philippines, Poland, Ecuador, Iran, Guatemala, Holand-Netherlands, Taiwan, Nicaragua, France, Jamaica, Scotland, Yugoslavia, Hong Kong, Trinadad&Tobago, Greece, Cambodia, Ireland.
citizenship: Native- Born in the United States, Foreign born- Not a citizen of U S , Native- Born in Puerto Rico or U S Outlying, Native- Born abroad of American Parent(s), Foreign born- U S citizen by naturalization.
own business or self employed: 0, 2, 1.
fill inc questionnaire for veteran's admin: Not in universe, Yes, No.
veterans benefits: 0, 2, 1.
weeks worked in year: continuous.
year: 94, 95.
income: - 50000, 50000+.
'''

class UCI_plus(torch.utils.data.Dataset):
    urls = [
        Path('Data/UCI/census-income.data.gz').resolve(),
        Path('Data/UCI/census-income.test.gz').resolve()
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pth'
    test_file = 'test.pth'

    

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()


        elif self.download == False:
            print("loading data. Already downloaded")
            # Load data
            folder_name = self.root
            
            training_file = folder_name / 'training.pth'
            with open(training_file, 'rb') as f:
                training_set = torch.load(f)
                
            test_file = folder_name / 'test.pth'
            with open(test_file, 'rb') as f:
                test_set = torch.load( f)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if train:
            self.data, self.labels = torch.load(
                self.root / self.processed_folder /self.training_file)
        else:
            self.data, self.labels = torch.load(
                self.root / self.processed_folder / self.test_file)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (self.root / self.processed_folder / self.training_file).is_file() and \
            (self.root / self.processed_folder / self.test_file).is_file()

    def download(self):
        if self._check_exists():
            return
        
        elif self.download == False:
            print("loading data. Already downloaded")
            # Load data
            folder_name = self.root
            
            training_file = folder_name / 'training.pth'
            with open(training_file, 'rb') as f:
                training_set = torch.load(f)
                
            test_file = folder_name / 'test.pth'
            with open(test_file, 'rb') as f:
                test_set = torch.load( f)

        else:
            # download files
            (self.root / self.raw_folder).mkdir(parents=True, exist_ok=True)
            (self.root / self.processed_folder).mkdir(parents=True, exist_ok=True)

        
            """for url in self.urls:
                print('Downloading ' + url)

                if isinstance(url, Path) and url.exists():
                    # Local file
                    with open(url, 'rb') as f:
                        data = f.read()
                else:
                    # Remote URL
                    with urllib.request.urlopen(str(url)) as response:
                        data = response.read()

                #data = urllib.request.urlopen(url)
                filename = url.rpartition('/')[2]
                file_path = self.root / self.raw_folder / filename
                with open(file_path, 'wb') as f:
                    f.write(data.read())
                with open(self.root / self.raw_folder / '.'.join(filename.split('.')[:-1]), 'wb') as out_f, \
                        gzip.GzipFile(file_path) as zip_f:
                    out_f.write(zip_f.read())
                os.unlink(file_path)"""
                
            for file_path in self.urls:
                print(f'Processing {file_path}')
                filename = file_path.name
                local_file_path = self.root / self.raw_folder / filename

                # Extract the file
                with open(local_file_path, 'rb') as f_in, \
                    gzip.GzipFile(fileobj=f_in) as zip_f, \
                    open(self.root / self.raw_folder / filename.replace('.gz', ''), 'wb') as f_out:
                    f_out.write(zip_f.read())
                    
                    
            # process and save as torch files
            print('Processing...')

            name_dict = OrderedDict()
            property_list = []
            for line in uci_info.split('\n'):
                if not line:
                    continue
                name, values = line.strip()[:-1].split(': ')
                name_dict[name] = []
                if values in ('ignore', 'label', 'continuous'):
                    pp = values
                else:
                    pp = 'normal'
                property_list.append(pp)

            self.uci_preprocess(self.root / self.raw_folder / 'census-income.data', name_dict, property_list)
            self.uci_preprocess(self.root / self.raw_folder / 'census-income.test', name_dict, property_list)

            for i, (name, values) in enumerate(name_dict.items()):
                value_set = list(sorted(list(set(values))))
                value_dict = dict()
                for j, value in enumerate(value_set):
                    value_dict[value] = j
                name_dict[name] = value_dict

            training_set = self.uci_process(self.root / self.raw_folder / 'census-income.data', name_dict, property_list)
            test_set = self.uci_process(self.root / self.raw_folder / 'census-income.test', name_dict, property_list)

            with open(self.root / self.processed_folder / self.training_file, 'wb') as f:
                torch.save(training_set, f)
            with open(self.root / self.processed_folder / self.test_file, 'wb') as f:
                torch.save(test_set, f)
                
            # Resolve the folder path
            folder_name = self.root
            # Create the new folder if it doesn't exist
            os.makedirs(folder_name, exist_ok=True)

            # Save the training set
            training_file = folder_name / 'training.pth'
            with open(training_file, 'wb') as f:
                torch.save(training_set, f)

            # Save the test set
            test_file = folder_name / 'test.pth'
            with open(test_file, 'wb') as f:
                torch.save(test_set, f)

            print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def uci_preprocess(path, name_dict, property_list):
        with open(path, 'r') as f:
            raw_data = f.readlines()
        for line in raw_data:
            if len(line.strip()) == 0:
                continue
            words = line.strip()[:-1].split(', ')
            if len(words) != 42:
                continue

            # make list
            for word, pp, (name, l) in zip(words, property_list, name_dict.items()):
                word = word.strip()
                if pp == 'continuous':
                    word = float(word)
                l.append(word)

    @staticmethod
    def uci_process(path, name_dict, property_list):
        with open(path, 'r') as f:
            raw_data = f.readlines()

        
        race_to_index = {
            'White': 0,
            'Black': 1,
            'Amer Indian Aleut or Eskimo': 2,
            'Asian or Pacific Islander': 3,
            'Other': 4}

        images = []
        labels = []
        for line in raw_data:
            if len(line.strip()) == 0:
                continue
            words = line.strip()[:-1].split(', ')
            if len(words) != 42:
                continue

            # make list
            image = []
            label = [None, None, None, None, None]  # age, education, marital stat, race, sex
            for word, pp, (name, values) in zip(words, property_list, name_dict.items()):
                word = word.strip()
                if pp == 'continuous':
                    word = float(word)
                    image.append(word)
                elif pp == 'ignore':
                    continue
                elif pp == 'label':
                    if name == 'education':
                        label[1] = int(word.startswith(('Bachelors', 'Some', 'Masters', 'Asso', 'Doctorate', 'Prof')))
                    elif name == 'marital stat':
                        label[2] = int(word == 'Never married')
                    #elif name == 'race':
                    #    label[3] = int(word == 'White')
                    elif name == 'race':
                        label[3] = race_to_index[word]
                    elif name == 'sex':
                        label[4] = int(word == 'Male')
                    else: # age
                        label[0] = int(float(word) >= 40)
                else:
                    # normal
                    one_hot = np.zeros(len(values))
                    one_hot[values[word]] = 1
                    image.append(one_hot)

            images.append(torch.Tensor(np.hstack(image)))
            labels.append(torch.LongTensor(label))
        return images, labels