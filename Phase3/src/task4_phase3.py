import numpy as np
import Distance_Similarity_Measure
from scipy.spatial import distance
from collections import defaultdict

class LSH:
    total_memory = 0

    def generate_random_vector(self, rows, col):
        return np.random.randn(rows, col)

    def create_hash_index(self, train_data, k):
        rand_hash_vector = self.generate_random_vector(len(train_data[0]), k)
        train_data = np.array(train_data)
        partition_info = np.dot(train_data, rand_hash_vector) >= 0
        # print(partition_info)
        binary_bucket_list = []
        for idx, binaryArr in enumerate(partition_info.astype(int).astype(str)):
            binary_val = ""
            for val in binaryArr:
                binary_val += val
            binary_bucket_list.append(binary_val)
        # print(binary_bucket_list)
        bucket_mapping = defaultdict(list)
        for idx, bucket in enumerate(binary_bucket_list):
            bucket_mapping[bucket].append(idx)
        # print(bucket_mapping)
        self.total_memory += rand_hash_vector.__sizeof__()
        self.total_memory += bucket_mapping.__sizeof__()
        hash_table = {'random_vectors' : rand_hash_vector, 'buckets' : bucket_mapping}
        return hash_table

    def create_layers(self, l, k, train_data):
        hash_layers = []
        total = 0
        for layer in range(l):
            hash_layer = self.create_hash_index(train_data, k)
            hash_layers.append(hash_layer)
            total += hash_layer.__sizeof__()
        # print("The total size of hash is : ", total)
        return hash_layers

    def get_total_index_size(self):
        return self.total_memory

    def test(self, n_words):
        Traindata = np.random.randn(n_words, 5)
        temp1 = np.random.randn(5, n_words)
        print(Traindata)
        print(temp1)
        binary_repr = Traindata.dot(temp1) >= 0
        # binary_repr1 = np.dot(Traindata, temp1) >= 0
        print(binary_repr)
        # print("Second matrix")
        # print(binary_repr1)
        # binary_inds = self.binary_2_integer(binary_repr)
        binary_string_arr = []
        for idx, binaryArr in enumerate(binary_repr.astype(int).astype(str)):
            print("idx", idx, "binary Arr", binaryArr)
            print("encoding ", str.encode(''.join(binaryArr)))
            binary_string_arr.append(str.encode(''.join(binaryArr)))
        print(binary_string_arr)
        table = defaultdict(list)
        for idx, bin_ind in enumerate(binary_string_arr):
            table[bin_ind].append(idx)
        print(table)
        hash_table = {'random_vectors': temp1, 'table': table}
        print("hash table")
        print(hash_table)

        print(binary_repr.astype(int).astype(str))


        return binary_string_arr

    def find_nearest_neighbor(self, query_image, hash_layers, t, max_allowed_distance):
        # print("query image ", query_image)
        # print("hashlayer data ", hash_layers)
        nearest_objects = set()
        current_hamming_distance = 0
        number_of_buckets = 0
        overall_images_considered = 0
        while(len(nearest_objects) < t):
            for hash in hash_layers:
                random_vector = np.array(hash['random_vectors'])
                # print("dimension of random vector ", random_vector.shape, "\nshape of query vector ", np.array(query_image).shape)
                # print("Random vector", random_vector)
                partition = np.dot(query_image, random_vector) >= 0
                bucket = ''
                # print("partition matrix", partition)
                for idx, binaryArr in enumerate(partition.astype(int).astype(str)):
                    binary_val = ""
                    for val in binaryArr:
                        binary_val += val
                # for p in partition.astype(int).astype(str):
                #     print("p ", p)
                #     bucket += p
                # print("bucket Name ", bucket)
                # print("Elemets present in the bucket ", hash['buckets'][bucket])

                if current_hamming_distance == 0:
                    temp = np.array(hash['buckets'][binary_val])
                    overall_images_considered += len(temp)
                    # print("temp", temp)
                    for element in temp:
                        nearest_objects.add(element)
                else:
                    for hash_table_bucket in hash['buckets'].keys():
                        xor = bin(int(binary_val, 2) ^ int(hash_table_bucket, 2))[2:]
                        hamming_distance = xor.encode().count(b'1')
                        # print("bitChanges ", hamming_distance)
                        if(current_hamming_distance == hamming_distance):
                            temp = np.array(hash['buckets'][hash_table_bucket])
                            overall_images_considered += len(temp)
                            # print("temp", temp)
                            for element in temp:
                                nearest_objects.add(element)

            current_hamming_distance += 1
            if(current_hamming_distance >= max_allowed_distance):
                break


                # nearest_objects.update(temp)
                # print(partition)
        print("hamming distance used", current_hamming_distance-1)
        # print("nearest objects : ", nearest_objects)
        print("overall images considered : ", overall_images_considered)
        print("unique images considered : ", len(nearest_objects))
        return nearest_objects

    def find_topK_nearest(self, data, query_image_vector, nearest_object_index_list):
        # print("coming inside the topk ")
        distance_dict = {}
        for i in nearest_object_index_list:
            dis = distance.euclidean(query_image_vector, data[i])
            distance_dict[i] = dis
        # print("before the closest objects are : ", distance_dict)
        distance_dict = dict(sorted(distance_dict.items(), key=lambda item: item[1]))
        return distance_dict

        # print("the closest objects are : ", distance_dict)





if __name__ == "__main__":

    obj = LSH()
    # obj.test(3)
    # k = int(input("Enter Number of hash functions in each Layer"))
    # l = int(input("Enter number of layers"))
    # vector_file = str(input("Enter the vector file name"))
    # feature_model = str(input("Enter feature model name"))
    # reduction = str(input("Do you want to apply reduction "))
    # reduction_method = ""
    # reduction_val = 0
    # if reduction == "yes":
    #     print("coming inside yes")
    #     reduction_method = str(input("enter feature model 1.PCA \n 2.SVD\n 3.LDA"))
    #     reduction_val = int(input("Enter the val of reduction k"))
    # queryName = str(input(print("enter query name")))
    # t = int(input("number of nearest neighbors want to find t :"))

    data = np.random.randn(4, 6)
    k = 5
    l = 3
    hash_layers = obj.create_layers(l, k, data)
    print("Data \n", data)
    # print(hash_layers)
    queryName = str(input(print("enter query name")))
    t= int(input("number of nearest neighbors want to find t :"))
    index = obj.find_index_of_query_image("testImage")
    nearest_object_index = obj.find_nearest_neighbor(data[index], index, hash_layers, 10, 2)
    obj.find_topK_nearest(data, index, nearest_object_index)




    # hash = obj.create_hash_index(data, 5)
    # print(hash)
    # print(hash['random_vectors'])