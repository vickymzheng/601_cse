import numpy
from scipy.spatial.distance import pdist, squareform
import math
from scipy.linalg import sqrtm
import random
from sklearn.decomposition import PCA as mypca
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import operator
#from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
#from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pdb

class GDataset( object):
    def __init__(self, gene_id, ground_truth, gene_exp):
        self.gene_id = gene_id
        self.ground_truth = ground_truth
        self.gene_exp = gene_exp

# read in txt file and record gene info and ground truth
def create_dataset(filename):
    global gene_num
    global exp_num
    
    if filename == "cho.txt":
        gene_num = 386
        exp_num = 16
    elif filename == "iyer.txt":
        gene_num = 517
        exp_num = 12
    elif filename == "small_data.txt":  # first 4 gene samples from cho.txt, just for testing
        gene_num = 4
        exp_num = 16

    elif filename == "small_data2.txt": # first 10 gene samples from cho.txt, just for testing
        gene_num = 10
        exp_num = 16

    elif filename == "easy_three_clusters.txt": # first 10 gene samples from cho.txt, just for testing
        gene_num = 35
        exp_num = 2

    elif filename == "new_dataset_1.txt": # first 10 gene samples from cho.txt, just for testing
        gene_num = 150
        exp_num = 4

    elif filename == "new_dataset_2.txt": # first 10 gene samples from cho.txt, just for testing
        gene_num = 6
        exp_num = 5



    id_list = []
    gt_list = []
    exp_matrix = numpy.zeros((gene_num,exp_num))

    line_num = 0
    with open(filename) as f:
            for line in f:
                parts = line.split('\t')
                
                id_list.append(float(parts[0]))
                gt_list.append(float(parts[1]))
                for i in range ( 2, exp_num+2):
                    exp_matrix[line_num][i-2] = float(parts[i])
                line_num += 1

    mydata = GDataset( id_list, gt_list, exp_matrix)
    return mydata


################################################# kmeans ##################################################
def calDis(x, center):
    dis = 0
    for i in range(3,len(x)):
        a = x[i] - center[i-3]
        b = math.pow(a,2)
        dis = dis + b
    dis = math.sqrt(dis)
    return dis

def Update(k,data, center):
    for i in range(0,len(data)):
        disArray = numpy.zeros(k)
        for j in range(0,len(center)):
            a = calDis(data[i], center[j])
            disArray[j] = a
        index = numpy.argmin(disArray)
        data[i][2] = index


def calCenter(k,data):
    sumArray = numpy.zeros([k,len(data[0])],dtype = float)
    sumSize = numpy.zeros(k)
    for i in range(0,len(data)):
        m = int(data[i][2])
        sumArray[m] = sumArray[m] + data[i]
        sumSize[m] = sumSize[m] + 1
    sumArray = numpy.delete(sumArray,[0,1,2],axis = 1)
    n = int(len(data[0]) - 3)
    center = numpy.zeros([k,n])
    for j in range(0,k):
        center[j] = sumArray[j] / sumSize[j]
    return center



def checkEql(x,y):
    flag = 0
    if numpy.array_equal(x,y):
        flag = 1
    return flag


def kmeans(k,InputData):
    center = numpy.zeros((k,len(InputData[0])))
    center_Index = random.sample(xrange(0, len(InputData)),k)
    for i in range(0,k):
        center[i] = InputData[center_Index[i]]
    
    center = numpy.delete(center,[0,1,2],axis = 1)


    flag = 0
    while flag == 0:
        lastC = InputData[:,2]
        Update(k,InputData,center)
        newC = InputData[:,2]
        center = calCenter(k,InputData)
        flag = checkEql(lastC,newC)
    return newC





############################################### h clustering ##############################################

def get_minpos(c_dist,n):

    cmin = c_dist[0][1]
    pos = [0,1]
    for i in range(0,n-1):
        for j in range(i+1,n):
            if cmin > c_dist[i][j]:
                pos = [i,j] # i < j
                cmin = c_dist[i][j]

    return pos



def compute_cdist(board_index,cluster_list, p_dist):
    #global p_dist
    
    n = len(board_index)
    board = numpy.zeros((n,n))
    
    for i in range(0,n-1):
        for j in range(i+1,n):
            tem = compute_cluster_pair_dist( cluster_list[board_index[i]], cluster_list[board_index[j]], p_dist)
            board[i][j] = tem
            board[j][i] = tem

    return board
            
                        
def compute_cluster_pair_dist( list1, list2, p_dist):   # min - single link
    #global p_dist
    list1_num = len(list1)
    list2_num = len(list2)
    
    min_dist = p_dist[list1[0]][list2[0]]
    for i in range(0,list1_num):
        for j in range(0,list2_num):
            tem = p_dist[list1[i]][list2[j]]
            if tem < min_dist:
                min_dist = tem
                        
    return min_dist
        
                        
                        
# clusters are in a list, every new formed the clusters push in: cluster_list
# each level stores the index of clusters: board_index
# put board_index at each level into a new list, the hc res: hc_res
def hierarchical_clustering(p_dist):
    
    #global ids
    #global gts
    #global ges
    global cluster_list
    #global hc_res
    #global p_dist
    #global board_index

    cluster_list = []
    for i in range(0,gene_num):
        cluster_list.append([i])

    board_index = range(0,gene_num)

    hc_res = []
    hc_res.append(range(0,gene_num))

    c_dist = p_dist
   
    new_index = -1
    new_cluster = []

    pos = []
    for round in range (1,gene_num):
        
        pos = get_minpos(c_dist,len(c_dist)) #pos[0] = row, pos[1] = col
        new_cluster = []
        new_cluster =cluster_list[board_index[pos[0]]] + cluster_list[board_index[pos[1]]]
        
        
        new_index = len(cluster_list)
        cluster_list.append(new_cluster)
        
       
        del board_index[pos[0]]
        del board_index[pos[1]-1]
        board_index.append(new_index)
        
    
        #this_res = []
        #this_res = board_index
        hc_res.append(board_index[:])   # if its hc_res.append(board_index), list elements changes with board_index

        c_dist = compute_cdist(board_index,cluster_list,p_dist) # not form in n*n matrix
   
    return hc_res


def hc_cluster_n(cluster_num,res):
    #global gene_num
    global cluster_list
    dims = numpy.matrix(res).shape
    gene_num = dims[1]
    res_n = res[gene_num-cluster_num]
    
    #labels = numpy.zeros((1,gene_num))
    labels = [0]*gene_num
    for i in range(1,cluster_num+1):
        print "cluster %d has genes:\n" % i
        #print [ x+1 for x in cluster_list[res_n[i-1]]]
        #print "\n\n"
        for x in cluster_list[res_n[i-1]]:
            print x+1
            #labels[0,x] = i
            labels[x] = i


    return labels


############################################## spectral k means clustering ###############################
def affinity_matrix(ges,sigma, gene_num):
    #global gene_num
    A = numpy.zeros((gene_num,gene_num))

    for i in range(0,gene_num-1):
        for j in range(i+1,gene_num):
            tem = numpy.array(ges[i]) - numpy.array(ges[j]) # ges[i], ges[j] may not right
            tem = numpy.sum(tem**2)
            tem = -tem / (2*sigma*sigma)
            a_res = math.exp( tem )
            A[i][j] = a_res
            A[j][i] = a_res

    return A


def norm_row(X, gene_num):
    #global gene_num
    Y = numpy.zeros((gene_num,gene_num))

    for i in range(0,gene_num):
        row_norm = math.sqrt( numpy.sum(numpy.array(X[i])**2) )
        Y[i,] = 1.0 * Y[i,] / row_norm
#        for j in range(0,cluster_num):
#            Y[i,j] = X[i,j]/row_norm

    return Y


def spectral_clustering(cluster_num,sigma,ids,gts,ges):
#    global ids
#    global gts
#    global ges

    dims = ges.shape
    gene_num = dims[0]


    A = affinity_matrix(ges,sigma, gene_num)
    D = numpy.zeros((gene_num,gene_num))

    for i in range(0,gene_num):
        D[i][i] = sum(A[i])

    A = numpy.matrix(A)
    D = numpy.matrix(D)
    D_msqrt =  numpy.matrix(numpy.linalg.inv( numpy.matrix(sqrtm(D)) ))
    #D_msqrt =  numpy.matrix(sqrtm( numpy.matrix(numpy.linalg.inv(D)) ))

    L = D_msqrt * A * D_msqrt


    w, v = numpy.linalg.eigh(L)
    X = v[:, range(gene_num-1, gene_num-cluster_num-1, -1)]
    Y = norm_row(X, gene_num)

    # do kmeans on Y here, each row of Y is a point
    tem_labels = [0]*gene_num
    ymatrix = numpy.column_stack((ids,gts,tem_labels,ges))
    kmeans_res = kmeans(cluster_num,ymatrix)
    #kmeans_res = KMeans(n_clusters = cluster_num).fit(Y)
    return kmeans_res

def choose_sigma( cluster_num,sigma_range,ids,gts,ges, try_time):
    #pdb.set_trace()
    sigma_choice_num = (numpy.matrix(sigma_range).shape)[1]
    all_external_index = [0] * sigma_choice_num
    tem_store = [0] * try_time
    
    for i in range(0,sigma_choice_num):
        tem_store = [0] * try_time
        for j in range(0,try_time):
            sp_labels = spectral_clustering(cluster_num,sigma_range[i],ids,gts,ges)
            sp_res = external_index(gts,sp_labels,1)
            tem_store[j] = sp_res

        all_external_index[i] = min(tem_store)

    print all_external_index
    #max_index = sigma_range.index(max(all_external_index))
    max_index, max_value = max(enumerate(all_external_index), key=operator.itemgetter(1))
    
    plt.plot(sigma_range,all_external_index)
    plt.title('rand index for different sigma')# give plot a title
    plt.xlabel('sigma')# make axis labels
    plt.ylabel('rand index')

    plt.show()

    return sigma_range[max_index]





############################################### validation #################################################

# function external_index computes external_index for clustering validation
# input: ground_truth, the list of ground truth labels for samples
#        clustering, the list of labels for samples produced by our algorithms
#        n, the number of samples
#        choice, if choice is 1, do rand index; else do Jaccard Coefficient
# output: the corresponding results of rand index or Jaccard Coefficient

def external_index(ground_truth, clustering, choice):
    ground_truth = numpy.array(ground_truth)
    clustering = numpy.array(clustering)
    choose_index = numpy.where(ground_truth != -1)[0]
    ground_truth = ground_truth[choose_index]
    clustering = clustering[choose_index]
    
    n = len(ground_truth)
    M_gt = numpy.ones((n,n))
    M_clt = numpy.ones((n,n))
    res = 0

    for i in range(0,n-1):
        for j in range( i+1,n):
            if ground_truth[i] !=  ground_truth[j]:
                M_gt[i][j] = 0
                M_gt[j][i] = 0

            if clustering[i] != clustering[j]:
                M_clt[i][j] = 0
                M_clt[j][i] = 0
                


    M_add = M_gt + M_clt
    M_sub = M_gt - M_clt

    M11 = (M_add == 2).sum()
    M00 = (M_add == 0).sum()

    M10 = (M_sub == -1).sum()
    M01 = (M_sub == 1).sum()

    print 'M11: %d' % M11
    print 'M00: %d' % M00
    print 'M10: %d' % M10
    print 'M01: %d' % M01
    
    if choice == 1: # rand index
        res = 1.0 * ( M11+M00 ) / ( M11+M00+M10+M01 )
    
    else:   # jaccord coiffiecint
        res = 1.0 * M11 / ( M11+M10+M01 )

    return res

############################################# PCA visualization #############################################
def pca_visual(data,label,dim,algo): # data is sample_num * feature_num
    pdb.set_trace()
    pca = mypca(n_components=dim)
    data = numpy.matrix(data).T
    pca.fit(data)
    data_pca = pca.components_
    pdb.set_trace()
    fig = plt.figure()
    if algo  == 1:
        title = 'Kmeans PCA scatter results'
    elif algo == 2:
        title = 'Hierarchical clustering PCA scatter results'
    elif algo == 3:
        title = 'Spectral clustering PCA scatter results'
    elif algo == 4:
        title = 'Mapreduce kmeans clustering PCA scatter results'

    if dim == 3:
        ax = fig.gca(projection='3d')
        ax.scatter(data_pca[0,],data_pca[1,], data_pca[2,], c=label,marker='o',s=30)
        ax.set_title(title)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
    

    elif dim == 2:
        ax = fig.gca()
        ax.scatter(data_pca[0,],data_pca[1,], c=label,marker='o',s=30)
        ax.set_title(title)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
    
    
    plt.grid()
    plt.show()
    return pca




############################################ function test ##################################################

## test funtion external_index with the example in slide- Clustering1: page 41
res1 = external_index([1,1,2,2,2],[1,1,1,2,2],1)
res2 = external_index([1,1,2,2,2],[1,1,1,2,2],2)
print res1
print res2
#pdb.set_trace()
## test funtion external_index ends here




############################################ main code starts here #########################################
# global variables, inited as follows
gene_num = 0
exp_num = 0
ids = []
gts= []
ges = []
cluster_list = []

###### things need to change for different experiments #########
filename = "cho.txt"  # input filename
#filename2 = "new_dataset_1.txt"
cluster_num = 2               # number of clusters
sigma = 2                       # spectral clustering config

# pca_dim = 2 # 2 for 2D visual; 3 for 3D visual
######


mydata = create_dataset(filename)
ids = mydata.gene_id
gts = mydata.ground_truth
ges = mydata.gene_exp
gene_num1 = ges.shape[0]
p_dist = pdist(ges, 'euclidean')
p_dist = squareform( p_dist )

#mydata2 = create_dataset(filename2)
#ids2 = mydata2.gene_id
#gts2 = mydata2.ground_truth
#ges2 = mydata2.gene_exp
#gene_num2 = ges2.shape[0]



k_cluster_num = 5
h_cluster_num = 2
s_cluster_num = 5
######################## k means ###############################
tem_labels = [0]*gene_num1
ymatrix = numpy.column_stack((ids,gts,tem_labels,ges))
k_labels = kmeans(k_cluster_num,ymatrix)
print k_labels

k_res1 = external_index(gts,k_labels,1)
print 'rand index of kmeans is %f\n' % k_res1
k_res2 = external_index(gts,k_labels,2)
print 'jc of kmeans is %f\n' % k_res2

k_pca = pca_visual(ges,k_labels,2,1)
pdb.set_trace()
###################### hc clustering ###########################
all_h = hierarchical_clustering(p_dist)
#print cluster_list
#print "\n\n"
hc_labels = hc_cluster_n(h_cluster_num,all_h)
print hc_labels

hc_res1 = external_index(gts,hc_labels,1)
print 'rand index of hc is %f\n' % hc_res1
hc_res2 = external_index(gts,hc_labels,2)
print 'jc of hc is %f\n' % hc_res2

hc_pca = pca_visual(ges,hc_labels,2,2)
#pdb.set_trace()

##################### spectral k means clustering ##############
sigma_range = numpy.arange(2,3,0.2)
sigma_opt = choose_sigma( s_cluster_num,sigma_range,ids,gts,ges,10)
print sigma_opt

sp_labels = spectral_clustering(s_cluster_num,sigma_opt,ids,gts,ges) # or sigma opt
print sp_labels

sp_res1 = external_index(gts,sp_labels,1)
print 'rand index of sp is %f\n' % sp_res1
sp_res2 = external_index(gts,sp_labels,2)
print 'jc of sp is %f\n' % sp_res2
sp_pca = pca_visual(ges,sp_labels,3,3)
pdb.set_trace()

