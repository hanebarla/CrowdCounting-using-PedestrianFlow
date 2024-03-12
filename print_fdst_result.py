import os
import csv

penalty = 0
dirs = []

for i in range(100):
    if (i+1) % 5 == 4 or (i+1) % 5 == 0:
        dirs.append(i+1)
print(dirs)
raise ValueError
root = "/groups1/gca50095/aca10350zi/habara_exp/FDST_{}_add".format(penalty)

ffs = ["noFF_Demo", "StaticFF_Demo", "DynamicFF_Demo", "BothFF_Demo"]

for ff in ffs:
    print(ff)
    output_path_dict = {}
    for d in dirs:
        scene_dir = os.path.join(root, str(d))
        # print(scene_dir)

        # no_ff = "BothFF_Demo  DynamicFF_Demo  noFF_Demo  StaticFF_Demo"
        with open(os.path.join(scene_dir, ff, "test.csv"), "r") as f:
            reader = csv.reader(f)
            l = [row for row in reader]
            print(d,*l[0])

    output_path_dict = {}
    for d in dirs:
        scene_dir = os.path.join(root, str(d))
        # print(scene_dir)

        # no_ff = "BothFF_Demo  DynamicFF_Demo  noFF_Demo  StaticFF_Demo"
        with open(os.path.join(scene_dir, ff, "test_pix.csv"), "r") as f:
            reader = csv.reader(f)
            l = [row for row in reader]
            print(d,*l[0])
