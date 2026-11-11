'''
what is yaml?
A human-readable file format used primarily for configuration files (like settings, database credentials, or app parameters).

pip install pyyaml

'''
import yaml

with open("first_config.yaml",'r') as file:
    config = yaml.safe_load(file)

print(config)
print("_" * 60)
print(config["my_app_name"])
print("_" * 60)
print(config["debug_mode"])
print("_" * 60)
print(config["database"])
print("_" * 60)
db_config = config["database"]
print(db_config["port"])
print("_" * 60)

print(config["features"])
print("_" * 60)