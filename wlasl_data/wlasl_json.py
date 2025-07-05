import math
import os
import sys
import json

def open_json(path):
    with open(path, 'r') as data:
        return json.load(data)
    
def reformat(data, subset=-1):
    labels = []
    database = {}
    if not subset == -1:
        for i in range(subset):
            item = data[i]
            label = item['gloss']
            if label not in labels: # add label to label list if its not there yet
                labels.append(label)
            total = len(item['instances']) # get number of label instances
            thresh = int(math.floor(total * .7)) # get %70 threshold
            count = 0
            for instance in item['instances']:
                entry_data = {}
                annotations = {}
                annotations['label'] = label
                
                if(count < thresh):
                    entry_data['subset'] = 'training'
                    entry_data['annotations'] = annotations
                else:
                    entry_data['subset'] = 'validation'
                    entry_data['annotations'] = annotations
                
                database[f'{instance['video_id']}'] = entry_data
                count += 1
                
        out_data = {}
        out_data['labels'] = labels
        out_data['database'] = database  
    else:
        for item in data:
            label = item['gloss']
            if label not in labels: # add label to label list if its not there yet
                labels.append(label)
            total = len(item['instances']) # get number of label instances
            thresh = int(math.floor(total * .8)) # get %70 threshold
            count = 0
            for instance in item['instances']:
                entry_data = {}
                annotations = {}
                annotations['label'] = label
                
                if(count < thresh):
                    entry_data['subset'] = 'training'
                    entry_data['annotations'] = annotations
                else:
                    entry_data['subset'] = 'validation'
                    entry_data['annotations'] = annotations
                
                database[f'{instance['video_id']}'] = entry_data
                count += 1
                
        out_data = {}
        out_data['labels'] = labels
        out_data['database'] = database
        
    return out_data
            
if __name__ == "__main__":
    json_path = sys.argv[1]
    out_path = sys.argv[2]
    if len(sys.argv) > 3:
        subset = int(sys.argv[3])
    else:
        subset = -1
    json_data = open_json(json_path)
    output = reformat(json_data, subset)
    with open(out_path, 'w') as dest:
        json.dump(output, dest)