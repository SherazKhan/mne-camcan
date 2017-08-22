import os
import glob
subject = 'AC077'

labels_path = '/autofs/cluster/transcend/sheraz/rs/452/AC077/*.label'
log_file = '/autofs/cluster/transcend/sheraz/rs/452/AC077/log.log'

labels = glob.glob(labels_path)

area = []
for label in labels:
    command = 'mris_anatomical_stats -l ' + label + ' ' + subject + ' ' + label[-8:-6] + ' white >& ' + log_file
    status = os.system(command)

    if not bool(status):
        datafile = file(log_file)
        for line in datafile:
            if 'total surface area' in line:
                area.append(line.split(' ')[-2])
        else:
            datafile.close()

