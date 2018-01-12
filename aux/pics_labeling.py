import os
path = "/home/experimentality/Documents/Degree work/Software/employee_classifier/Allowed/"
files = os.listdir(path)
i = 1

for file in files:

    orig_name = file
    os.rename(os.path.join(path, file), os.path.join(path, '0.allowed' + str(i) + '.png'))
    i = i + 1
