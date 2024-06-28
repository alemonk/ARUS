import os

num_poses = 382
distance_cm = 0.045

filename = 'recon/img_pose.txt'
# Remove the files if they exist
if os.path.exists(filename):
    os.remove(filename)

# Define initial pose values
initial_x = 0
initial_y = 0
initial_z = 0
qx = 0
qy = 0
qz = 0
qw = 1

# Open the new img_pose.txt file for writing
with open(filename, 'w') as file:
    for i in range(num_poses):

        # Calculate the new x-coordinate for each image
        x = initial_x
        y = initial_y
        z = initial_z + i * distance_cm

        # Write the new pose line to the file
        file.write(f'{x} {y} {z} {qx} {qy} {qz} {qw}\n')

print(f"{filename} file has been created successfully.")
