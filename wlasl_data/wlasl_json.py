import os
import sys
import json

def open_json(path):
    with open(path, 'r') as data:
        return json.load(data)
    
def reformat(data):
    labels = []
    database = {}
    for item in data:
        label = item['gloss']
        if label not in labels:
            labels.append(label)
        for entry in item['instances']:
            database[f'{entry['video_id']}'] = label
    out_data = {}
    out_data['labels'] = labels
    out_data['database'] = database
    return out_data
            
if __name__ == "__main__":
    json_path = sys.argv[1]
    out_path = sys.argv[2]
    json_data = open_json(json_path)
    output = reformat(json_data)
    with open(out_path, 'w') as dest:
        json.dump(output, dest)