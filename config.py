import json

config = {
    "project_name": "Ulukhaktok", # Name of the project that will be used in output files' name
    "date_position": (15,23), # Starting and ending index positions of the date in filename 
    "time_position": (24,30), # Starting and ending index positions of the time in filename 
    "channel_number": 1, # Channel number to be used when audio is sterio 
    "min_freqs": [50],  # lower end frequencies for analysis
    "max_freqs": [1000],  # upper end frequencies for analysis
    "segment_length": 100,  # Duration of each audio segment in seconds
    "system_noise_frequencies": [375, 132, 131, 750, 220, 75, 440, 660, 1124, 125],  # Frequencies to mask out
}

# Path to the JSON file
json_file_path = 'config.json'

# Save the configuration data to a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(config, json_file, indent=4)  # Write the `config` dictionary to the JSON file with indentation for readability

