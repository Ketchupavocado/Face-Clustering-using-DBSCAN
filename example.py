import get_encodings as enc
import face_cluster as cluster
#Time the code
import time


root_path = 'Timer/'

# Create a face database from existing faces
dir_name, data_len = enc.create_face_database(root_path)


start_time = time.time()
unq_faces, res_dir = cluster.do_cluster(dir_name)
print("Number of unique face detected: ", unq_faces)
print(f'You can find clustered face in "{res_dir}" directory')
cluster_time = time.time() - start_time
print(f"Face clustering completed in {cluster_time:.2f} seconds.")
