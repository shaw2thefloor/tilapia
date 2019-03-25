# Created by fshaw at 06/03/2019
# split data out into species based on the spreadsheet of species and image id

import pandas as pd
import os, shutil

# Assign spreadsheet filename to `file`
file = '/home/fshaw/Documents/fish/Oreochromis_sequencing_summary_31Jul18_wrangled.xlsx'

# Load spreadsheet
xl = pd.ExcelFile(file)

original_images = "/home/fshaw/Documents/fish/tilapia_images"

input_dir = "/home/fshaw/Documents/fish/split/all"
output_dir = "/home/fshaw/Documents/fish/split/"

# create train and test dirs in split
train = os.path.join(output_dir, 'train')
test = os.path.join(output_dir, 'test')

# Load a sheet into a DataFrame by name: df1
df1 = xl.parse('All_Samples')

# get unique species names
unique = df1["FieldID/phenotype"].unique()

try:

    # delete and recreate dirs

    #shutil.rmtree(input_dir, ignore_errors=True)
    #shutil.rmtree(test, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(train)
    os.makedirs(test)
except OSError as e:
    print("error making directories: " + str(e))

dir_count = 0
for dir_name in unique:
    dir_count += 1
    if not pd.isnull(dir_name):
        os.makedirs(os.path.join(input_dir, dir_name))
copy_count = 0
map = list()
log = open("log.txt", "w+")
print("Copying...")
for index, row in df1.iterrows():
    id = str(row["ID"])
    dirs = os.listdir(original_images)
    for f in dirs:
        if id in f:
            # is there is a match, copy to the dir matching the species
            log.write("id:- " + id + " file:- " + f + "\n")
            copy_count += 1
            shutil.copyfile(os.path.join(original_images, f), os.path.join(input_dir, row["FieldID/phenotype"], f))
print(str(copy_count) + " files copied into " + str(dir_count) + " directories")
log.close()
