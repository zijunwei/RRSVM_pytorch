import json
import sys, os

# when you're saving to json, make sure you convert dumps first then dump
def load_json(data_file):
    with open(data_file, 'r') as df:
        json_string = json.load(df)
        data = json.loads(json_string)
    return data


def save_json(json_object, data_file):
    with open(data_file, 'w') as df:
        json_object_string = json.dumps(json_object)
        json.dump(json_object_string, df)


def load_list(data_file):
    with open(data_file, 'r') as df:
        loaded_list = [line.strip() for line in df]

    return loaded_list


def save_list(obj_list, data_file):
    with open(data_file, 'w') as df:
        for s_object in obj_list:
            df.write('{:s}\n'.format(s_object))


# Some useless:
#   # if os.path.isfile(image_name_file):
  #   with open(image_name_file, 'a') as f:
  #     for s_image_name in hit_image_names:
  #       f.write('{:s}\n'.format(s_image_name))
  # else:
  #   with open(image_name_file, 'w+') as f:
  #     for s_image_name in hit_image_names:
  #       f.write('{:s}\n'.format(s_image_name))
