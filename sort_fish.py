# Created by fshaw at 06/03/2019
# split data out into species based on the spreadsheet of species and image id, then split further into test and
# training data
from pathlib import Path
import pandas as pd
import os, shutil, math, random, stat

num_splits = 20

def on_rm_error( func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    try:
        os.chmod( path, stat.S_IWRITE )
        os.unlink( path )
    except FileNotFoundError as f:
        print(f)


# Assign spreadsheet filename to `file`
proj_dir = Path('C:/Users/fshaw/Documents/PycharmProjects/tilapia/')
data = proj_dir / "data"
ss_file = data / "Oreochromis_sequencing_summary_31Jul18_wrangled.xlsx"
images = data / "image_data"
raw = data / "raw"
# Load spreadsheet
xl = pd.ExcelFile(ss_file)

original_images = raw / "small_images"
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
    os.chmod(by_species_dir, 0o777)
except OSError as e:
    print("error making directories: " + str(e))

shutil.rmtree(by_species_dir, onerror=on_rm_error)

os.makedirs(by_species_dir)


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




for n in range(0, num_splits):
    print("fold: " + str(n))
    fold_dir = model_data / str(n)
    try:
        shutil.rmtree(fold_dir)
    except:
        pass
    # for each dir in input get n files and put in test, then put the rest in training
    for dir in os.listdir(by_species_dir):

        # choose m random indexes from the n species images where m = 1/3 * n
        file_list = os.listdir(by_species_dir / dir)
        num_img = len(file_list)
        ratio_of_test_data = 1 / 3
        num_idx = math.floor(num_img * ratio_of_test_data)
        test_idxs = list()
        ran = 0

        # loop to generate <num_idx> non-duplicated random integers in the interval [0, num_images]
        while len(test_idxs) < (num_idx):
            i = random.randint(0, num_img - 1)
            if i not in test_idxs:
                test_idxs.append(i)


        test_dir = model_data / str(n) / "test" / dir
        train_dir = model_data / str(n) / "train" / dir
        os.makedirs(test_dir)
        os.makedirs(train_dir)
        count = 0
        for f in file_list:

            if count in test_idxs:
                shutil.copy2(by_species_dir / dir / f, test_dir / f)

            else:
                shutil.copy2(by_species_dir / dir / f, train_dir / f)

            count = count + 1


'''
    # now iterate directories and delete those which have no images in them
    for path, dirs, files in os.walk(by_species_dir):
        for dir in dirs:
            for s_path, s_dirs, s_files in os.walk(os.path.join(path, dir)):

                if len(s_files) < 2:
                    shutil.rmtree(os.path.join(path, dir))
                    '''