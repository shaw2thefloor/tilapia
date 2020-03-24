# Created by fshaw at 06/03/2019
# split data out into species based on the spreadsheet of species and image id, then split further into test and
# training data
from pathlib import Path
import pandas as pd
import os, shutil, math, random

# Assign spreadsheet filename to `file`
proj_dir = Path('C:/Users/fshaw/OneDrive - Norwich BioScience '
                'Institutes/dev/tilapia/')
data = proj_dir / "data"
ss_file = data / "Oreochromis_sequencing_summary_31Jul18_wrangled.xlsx"
images = data / "images"
# Load spreadsheet
xl = pd.ExcelFile(ss_file)

original_images = images / "small_images"
by_species_dir = images / "split" / "all"

# create train and test dirs in split
model_data = images / "split"
train = model_data / 'train'
test = model_data / 'test' 
# Load a sheet into a DataFrame by name: df1
df1 = xl.parse('All_Samples')

# get unique species names
unique = df1["FieldID/phenotype"].unique()

try:
    # delete and recreate dirs
    shutil.rmtree(by_species_dir, ignore_errors=True)
    shutil.rmtree(model_data, ignore_errors=True)
    os.makedirs(by_species_dir)
    os.makedirs(train)
    os.makedirs(test)
except OSError as e:
    print("error making directories: " + str(e))

dir_count = 0
for dir_name in unique:
    dir_count += 1
    if not pd.isnull(dir_name):
        os.makedirs(os.path.join(by_species_dir, dir_name))
copy_count = 0
map = list()

for index, row in df1.iterrows():
    id = str(row["ID"])
    dirs = os.listdir(original_images)
    flag = True
    for f in dirs:

        if id in f:
            # is there is a match, copy to the dir matching the species
            log = open("log.txt", "w+")
            log.write("id:- " + id + " file:- " + f + "\n")
            log.close()
            copy_count += 1
            shutil.copyfile(os.path.join(original_images, f),
                            os.path.join(by_species_dir, row["FieldID/phenotype"], f))
            flag = False
log = open("log.txt", "w+")
log.write(str(copy_count) + " files copied into " + str(dir_count) + " directories")
log.close()

# now iterate directories and delete those which have no images in them
for path, dirs, files in os.walk(by_species_dir):
    for dir in dirs:
        for s_path, s_dirs, s_files in os.walk(os.path.join(path, dir)):

            if s_path.endswith("Not Given") or s_path.endswith("Hybrid"):
                pass
                #shutil.rmtree(os.path.join(path, dir))
            elif len(s_files) > 10:
                pass

            else:
                pass
                #shutil.rmtree(os.path.join(path, dir))


# for each dir in input get n files and put in test, then put the rest in training
for idx, dir in enumerate(os.walk(by_species_dir)):

    # get dir name
    temp_dir_name = dir[0]
    out_test_name = temp_dir_name.replace('all', 'test')
    out_train_name = temp_dir_name.replace('all', 'train')
    shutil.rmtree(out_test_name, ignore_errors=True)
    shutil.rmtree(out_train_name, ignore_errors=True)
    os.mkdir(out_test_name)
    os.mkdir(out_train_name)

    # choose m random indexes from the n species images where m = 1/3 * n
    num_img = len(dir[2])
    ratio_of_test_data = 1 / 3
    num_idx = math.floor(num_img * ratio_of_test_data)
    test_idxs = list()
    for i in range(0, num_idx - 1):
        test_idxs.append(random.randint(0, num_img - 1))

    for idx_1, f in enumerate(dir[2]):
        if idx_1 in test_idxs:
            shutil.copyfile(os.path.join(dir[0], f), os.path.join(out_test_name, f))
        else:
            shutil.copyfile(os.path.join(dir[0], f), os.path.join(out_train_name, f))
