# Created by fshaw at 28/02/2019
# split data into test and training sets
import os, shutil

# take directory of info and create:
#                         dir --
#                               |
#                               --train
#                               |
#                               --test

input_dir = "/home/fshaw/Documents/fish/all/"
output_dir = "/home/fshaw/Documents/fish/split/"

# create train and test dirs in split
train = os.path.join(output_dir, 'train')
test = os.path.join(output_dir, 'test')

try:
    # delete and recreate dirs
    shutil.rmtree(train, ignore_errors=True)
    shutil.rmtree(test, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(train)
    os.makedirs(test)
except OSError as e:
    print("error making directories: " + str(e))

# for each dir in input get n files and put in test, then put the rest in training
for idx, dir in enumerate(os.walk(input_dir)):
    if idx >= 0 and idx <= 120:
        # get dir name
        temp_dir_name = dir[0]
        out_test_name = temp_dir_name.replace('all', 'split/test')
        out_train_name = temp_dir_name.replace('all', 'split/train')
        shutil.rmtree(out_test_name, ignore_errors=True)
        shutil.rmtree(out_train_name, ignore_errors=True)
        os.mkdir(out_test_name)
        os.mkdir(out_train_name)

        for idx, f in enumerate(dir[2]):
            if idx < 20:
                shutil.copyfile(os.path.join(dir[0], f), os.path.join(out_test_name, f))
            else:
                shutil.copyfile(os.path.join(dir[0], f), os.path.join(out_train_name, f))
