import os, shutil
from PIL import Image, ImageFilter

input_dir = '/home/fshaw/Documents/fish/tilapia_images'
output_dir = '/home/fshaw/Documents/fish/small_images'
#shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir)

# x is target size
x = 500

out = open("log.txt", "w+")
for root, dirs, files in os.walk(input_dir):
    count = 0
    for file in files:
        if not file.startswith("."):
            im = Image.open(os.path.join(input_dir, file))
            w = im.size[0]
            # get scale factor
            f = w / x
            new_size = (int(im.size[0] * 1 / f), int(im.size[1] * 1 / f))
            resized_im = im.resize(new_size)
            resized_im.save(os.path.join(output_dir, file))

            #landscape = im.size[0] > im.size[1]
            #if not landscape:
            #    count += 1
                #im = im.rotate(90)
                #im.show()
            #    im.save(os.path.join(input_dir, file))
            #    out.write(str(file) + '\n')
    #out.write(str(count))
out.close()
