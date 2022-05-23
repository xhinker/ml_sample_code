import numpy as np
import random

def grouping(points,cps) -> dict:
    '''
    This function will group the points based on the center points(cps), and return
    result in a dictionary object
    {
        cluster_index: [(x1,y1),(x2,y2)]
        ...
    }
    '''
    clusters_dict = {}

    # init the empty dictionary
    for i in range(len(cps)):
        clusters_dict[i] = []

    for p in points:
        temp_distances = []
        for i,cp in enumerate(cps):
            # calculate the distance
            d = (p[0]-cp[0])**2 + (p[1]-cp[1])**2
            temp_distances.append(d)
        temp_distances = np.array(temp_distances)
        min_d_i = np.argmin(temp_distances)
        clusters_dict[min_d_i].append(tuple(p))
    return clusters_dict

def get_new_centers(cps:list,clusters_dict:dict):
    '''
    calculate the new centers, return is_renewed and a list of new center points
    When the caller sees is_renewed is 0, means the center points are converged and can 
    no longer find better center points. 
    '''
    is_renewed = 0
    for c_i in clusters_dict:
        p_num           = len(clusters_dict[c_i])
        x_d_sum,y_d_sum = 0,0
        for p in list(clusters_dict[c_i]):
            x_d_sum = x_d_sum + p[0]
            y_d_sum = y_d_sum + p[1]
        x_d_mean = x_d_sum//p_num if p_num!=0 else cps[c_i][0]
        y_d_mean = y_d_sum//p_num if p_num!=0 else cps[c_i][1]
        if abs(x_d_mean - cps[c_i][0])>0 or abs(y_d_mean - cps[c_i][1])>0:
            cps[c_i]=[x_d_mean,y_d_mean]
            is_renewed =1
    return is_renewed,cps

def get_distance_sum(cps:list,clusters_dict:dict):
    '''
    calculate the points' distance to their center points and return the sum of it
    the smaller of the sum distance, the better. 
    '''
    d_sum =0 
    for c_i in clusters_dict:
        for p in list(clusters_dict[c_i]):
            d = (p[0]-cps[c_i][0])**2 + (p[1]-cps[c_i][1])**2
            d_sum = d_sum + d
    return d_sum

def kmeans(points:list,k=3,n=10,max_iter=100):
    '''
    This function wrap up all steps together, give points and cluster number k. 
    It will return the best cluster centers and points associated with it. 

    * points: the input points
    * k: number of clusters
    * n: times to initial center points, the higher the n set, the higher chance to get the best cluster
    * max_iter: times to repeat for one initialized center points to. 
    '''
    points = np.array(points)
    x_l = points[:,0]
    y_l = points[:,1]
    k = k
    result_dict = {}
    for _ in range(n):
        # get x,y min and max
        min_x,max_x = min(x_l),max(x_l)
        min_y,max_y = min(y_l),max(y_l)

        # init cluster points
        cps = []
        for _ in range(k):
            cps.append([random.randint(min_x,max_x),random.randint(min_y,max_y)])

        # init the first clusters 
        new_c = grouping(points=points,cps=cps)
        iter_n = 0
        while iter_n<max_iter:
            is_renewed,cps = get_new_centers(cps,new_c)
            if not is_renewed:
                break
            new_c = grouping(points,cps)
            iter_n+=1
        
        # save the result
        result = (cps,new_c)
        result_dict[get_distance_sum(cps,new_c)]=result

    # find the result with least total distance
    key_list = [k for k in result_dict]
    min_key = min(key_list)
    return result_dict[min_key]


#%% run it 
import matplotlib.pyplot as plt 
points = [[2,3]
         ,[3,4]
         ,[1,2]
         ,[10,12]
         ,[12,10]
         ,[13,11]
         ,[10,3]
         ,[11,2]
         ,[12,4]]
points = np.array(points)
x_l = points[:,0]
y_l = points[:,1]

result = kmeans(points=points)
print(result)

#%% 
plt.scatter(x_l,y_l)
cps = np.array(result[0])
cps_x,cps_y = cps[:,0],cps[:,1]
plt.scatter(cps_x,cps_y)
