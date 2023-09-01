import numpy as np
import torch
from jakteristics import compute_features
from pytorch3d.ops import estimate_pointcloud_local_coord_frames, ball_query
from skimage.filters import threshold_otsu
from sklearn.linear_model import RANSACRegressor
from segmentation.plot_segmentation import visualize_points

#TODO move to the config file
device = 'cuda:0'
server = True

if not server:
    import matplotlib.pyplot as plt


def filter_normals(point_cloud:torch.Tensor,k_nn:int,angle_deg_tresh:float,ref_normal:torch.Tensor = torch.tensor([0,0,1],dtype=torch.float32),cpu = False):

    if cpu:
        point_cloud = point_cloud.cpu()

    #estimate pca
    knn_res = estimate_pointcloud_local_coord_frames(
        point_cloud[:,:3].unsqueeze(0).float(),neighborhood_size=k_nn
    )
    eigen_values,eigen_vectors = knn_res
    eigen_values = eigen_values.squeeze(0).to(device)
    eigen_vectors = eigen_vectors.squeeze(0).to(device)

    #get normal vectors
    normals = eigen_vectors[ :, :, 0]

    #compute the reference angles
    ref_normal = ref_normal.to(device)
    dot_products = torch.matmul(normals,ref_normal)
    cosines = dot_products / (torch.linalg.norm(normals,dim=1) * torch.linalg.norm(ref_normal))
    cosines = cosines.cpu()
    angles_deg = torch.rad2deg(torch.arccos(np.abs(cosines)))

    #get the valid mask
    angle_mask = angles_deg <= angle_deg_tresh

    if cpu:
        point_cloud = point_cloud.to(device)
        normals = normals.to(device)
        angles_deg = angles_deg.to(device)
        eigen_values = eigen_values.to(device)


    return angle_mask,angles_deg,normals,eigen_values

def segment_asphalt_naive(point_cloud:torch.Tensor, tresh_up:float,tresh_down:float,num_lasers:int,visualization = False):

    max_laser_id = point_cloud[:,6].max()
    laser_id_mask = point_cloud[:,6] >= max_laser_id - num_lasers
    sample_points = point_cloud[laser_id_mask,:].cpu()

    min_intensity = sample_points[:,3].min().to(int).item()
    max_intensity = sample_points[:,3].max().to(int).item()

    hist,_ = torch.histogram(sample_points[:,3].float(),bins= max_intensity - min_intensity,range=(min_intensity,max_intensity))
    peak_idx = torch.argmax(hist)
    peak_intensity = min_intensity + peak_idx

    asphalt_int_min = peak_intensity - tresh_down
    asphalt_int_max = peak_intensity + tresh_up

    if visualization and not server:
        plt.figure()
        plt.plot(hist)
        plt.scatter([peak_idx],hist[peak_idx],color= 'red')
        plt.scatter([peak_idx - tresh_down,peak_idx + tresh_up],[hist[peak_idx - tresh_down],hist[peak_idx + tresh_up]],color='green')
        plt.show()

    asphalt_mask = torch.logical_and(point_cloud[:,3] <= asphalt_int_max , point_cloud[:,3] >= asphalt_int_min)
    asphalt_points = point_cloud[asphalt_mask,:]

    asphalt_mask = asphalt_mask.to(device)
    asphalt_points = asphalt_points.to(device)

    return asphalt_points,asphalt_mask

def get_lower_pca_variance(point_cloud:torch.Tensor,k_nn:int,eigen_values:torch.Tensor = torch.tensor([0]),cpu = False):

    if cpu:
        point_cloud = point_cloud.cpu()

    if eigen_values.shape[0] <= 1:
        #compute the eigenvalues
        knn_res = estimate_pointcloud_local_coord_frames(
        point_cloud[:,:3].unsqueeze(0).float(),neighborhood_size=k_nn
        )
        eigen_values,_ = knn_res
        eigen_values.squeeze_(0)
    
    if cpu:
        point_cloud = point_cloud.to(device)

    eigen_values = eigen_values.to(device)
    point_cloud = point_cloud.to(device)
    #compute the variance of the lowest eigenvalues
    eigen_values,_ = torch.sort(eigen_values,dim=1)
    var = (eigen_values[:,1] + eigen_values[:,0]) / (eigen_values[:,0] * eigen_values[:,1] + 1e-10)

    return var

def get_mask(subset:torch.Tensor,source:torch.Tensor):
    b = subset.cpu().numpy()
    a = source.cpu().numpy()
    baseval = np.max([a.max(), b.max()]) + 1
    n_cols = a.shape[1]
    a = a * baseval ** np.array(range(n_cols))
    b = b * baseval ** np.array(range(n_cols))
    c = np.isin(np.sum(a, axis=1), np.sum(b, axis=1))

    c = torch.from_numpy(c).to(device)

    return c
    
def extend_invalid_mask(point_cloud:torch.Tensor,valid_mask:torch.Tensor,radius:float,cpu:bool = False):
    if cpu:
        point_cloud = point_cloud.cpu()
        valid_mask = valid_mask.cpu()

    invalid_points = point_cloud[~valid_mask,:3].unsqueeze(0).float()
    valid_points = point_cloud[valid_mask,:3].unsqueeze(0).float()

    if not cpu:
        invalid_points = invalid_points.to(device)
        valid_points = valid_points.to(device)

    b_query = ball_query(invalid_points,valid_points,radius=radius,return_nn=False)
    invalid_indices = b_query[1].squeeze(0).to(device)
    flat_invalid_indices = invalid_indices[invalid_indices != -1]

    if cpu:
        valid_mask = valid_mask.to(device)
        point_cloud = point_cloud.to(device)

    new_valid_mask = valid_mask.clone()
    new_valid_mask[new_valid_mask][flat_invalid_indices] = 0

    return new_valid_mask

def get_asphalt_pca_variance(point_cloud_variance:torch.Tensor,asphalt_mask:torch.Tensor,hist_density:float):

    #compute variance
    reference_variance = point_cloud_variance[asphalt_mask].cpu()

    #get the biggest peak
    ref_range = reference_variance.max() - reference_variance.min()
    bin_count = torch.floor(ref_range / hist_density).item()
    hist,_ = torch.histogram(reference_variance.float(),bins=int(bin_count),range=(reference_variance.min().float().item(),reference_variance.max().float().item()))
    max_peak = torch.argmax(hist)
    max_peak_value = max_peak / bin_count * ref_range + reference_variance.min()

    return max_peak_value

def get_asphalt_verticality(point_cloud_verticality:torch.Tensor,asphalt_mask:torch.Tensor,hist_density:float):

    #treshold to remove influence of obvious outliers
    valid_verticallity_mask = point_cloud_verticality[asphalt_mask] <= -1

    #compute verticality
    reference_verticality = point_cloud_verticality[asphalt_mask][valid_verticallity_mask].cpu()

    #get the biggest peak
    ref_range = reference_verticality.max() - reference_verticality.min()
    bin_count = torch.floor(ref_range / hist_density).item()
    hist,_ = torch.histogram(reference_verticality.float(),bins = int(bin_count),range=(reference_verticality.min().item(),reference_verticality.max().item()))
    max_peak = torch.argmax(hist)
    max_peak_value = max_peak / bin_count * ref_range + reference_verticality.min()

    return max_peak_value
    
def plane_fit_refine(asphalt_points:torch.Tensor,lane_candidates:torch.Tensor,fit_inlier_tresh:float,filter_inlier_tresh:float,valid_angle_tresh:float,normals:torch.Tensor):

    X = np.concatenate([asphalt_points[:,:2].cpu().numpy(),lane_candidates[:,:2].cpu().numpy()],axis=0)
    y = np.concatenate([asphalt_points[:,2].cpu().numpy(),lane_candidates[:,2].cpu().numpy()],axis=0)

    regressor = RANSACRegressor(residual_threshold=fit_inlier_tresh)
    regressor.fit(X,y)

    a,b = regressor.estimator_.coef_
    c = -1
    d = regressor.estimator_.intercept_

    normal_vec = torch.tensor([a,b,c],dtype=torch.float32)
    normal_vec /= torch.linalg.norm(normal_vec)

    distances = torch.abs((a*lane_candidates[:,0] + b*lane_candidates[:,1] + c*lane_candidates[:,2]+d) / torch.sqrt(torch.tensor([a**2 + b**2 + c**2],dtype=torch.float32)))
    inlier_mask = distances <= filter_inlier_tresh

    #check the fit validity with the normals
    dot_products = torch.matmul(normals[inlier_mask,:].cpu(),normal_vec)
    cosines = dot_products / (torch.linalg.norm(normals[inlier_mask,:].cpu(),dim=1) * torch.linalg.norm(normal_vec))
    angles_deg = torch.rad2deg(torch.arccos(np.abs(cosines)))
    if torch.mean(angles_deg) >= valid_angle_tresh:
        print(f'invalid RANSAC fit: {torch.mean(angles_deg).item()}')
        return torch.ones_like(lane_candidates[:,0],dtype=bool).to(device)
    else:
        return inlier_mask.to(device)

def pipeline(point_cloud:torch.Tensor,angle_deg_tresh:float):
    
    #TODO constrain adaptive otsu treshold
    #TODO use more spatial features
    #TODO ball_query + similarity -> add false negatives

    final_mask = None

    #filter normals angles
    angle_mask,angles,normals,_ = filter_normals(point_cloud,20,angle_deg_tresh,cpu=False)
    angle_mask = extend_invalid_mask(point_cloud,angle_mask,0.3,cpu=False)
    final_mask = angle_mask.clone().to(device)
    print('normals computed')

    #segment asphalt points
    asphalt_points,asphalt_mask = segment_asphalt_naive(point_cloud[angle_mask,:],3,15,20,visualization=True) #!segmenting attempt 1

    #compute verticality and its asphalt treshold
    print(torch.isnan(point_cloud[angle_mask,:3]).any())
    point_cloud_verticality = compute_features(point_cloud[angle_mask,:3].cpu().numpy().astype(np.float64),search_radius=0.1,feature_names=['verticality']) #!which points to compute from?

    point_cloud_verticality[np.isnan(point_cloud_verticality)] = 0
    point_cloud_verticality = torch.from_numpy(point_cloud_verticality).to(device)
    point_cloud_verticality = torch.log(point_cloud_verticality + 1e-13)

    asphalt_verticality = get_asphalt_verticality(point_cloud_verticality,asphalt_mask,0.1)
    eps = 2
    verticality_treshold = asphalt_verticality + eps
    print(f'asphalt verticality is {asphalt_verticality} and treshold is {verticality_treshold}')
    verticality_mask = point_cloud_verticality <= verticality_treshold

    final_mask[angle_mask] = verticality_mask.view(-1)

    #compute pca variance and its asphalt treshold
    point_cloud_variance = get_lower_pca_variance(point_cloud[final_mask,:3],50,cpu=False)
    point_cloud_variance = torch.log(torch.abs(point_cloud_variance) + 1e-13)
    asphalt_points,asphalt_mask = segment_asphalt_naive(point_cloud[final_mask,:],20,25,20,visualization=True) #!segmenting attempt 2
    asphalt_variance = get_asphalt_pca_variance(point_cloud_variance,asphalt_mask,0.1)
    eps = 2
    variance_treshold = asphalt_variance - eps
    variance_mask = point_cloud_variance >= variance_treshold
    variance_mask = ~extend_invalid_mask(point_cloud[final_mask,:3],~variance_mask,0.4,cpu = False)
    final_mask[final_mask.clone()] = variance_mask.view(-1).to(device)
    print(f'asphalt variance is {asphalt_variance} and treshold is {variance_treshold}')


    #visualize_points(point_cloud[~final_mask,:].cpu().numpy())

    #treshold the instensities
    intensities = point_cloud[final_mask,3].cpu().numpy()
    otsu_treshold = threshold_otsu(intensities,nbins=255)
    print(f'intensity treshold: {otsu_treshold}')
    if not server:
        plt.hist(intensities,bins=255,range=[0,255])

    #visualize_points(point_cloud[final_mask,:].cpu().numpy)

    intensity_mask = intensities >= otsu_treshold
    intensity_mask = torch.from_numpy(intensity_mask).to(device)
    final_mask[final_mask.clone()] = intensity_mask

    #RANSAC refinement
    print('refining mask')
    refined_mask = plane_fit_refine(
        asphalt_points.cpu(),
        point_cloud[final_mask,:].cpu(),
        0.05,0.05,10,normals[final_mask,:])\
        
    refined_final_mask = final_mask.clone()
    refined_final_mask[final_mask] = refined_mask

    return final_mask,refined_final_mask

def pipeline_frames(point_cloud:torch.Tensor,min_frame:int,max_frame:int):

    final_mask = torch.zeros_like(point_cloud[:,0],dtype=bool).to(device)
    final_mask_refined = final_mask.clone()
    print('mask initialized')

    for frame_id in range(min_frame,max_frame + 1):
        print(f'processing frame {frame_id}')

        frame_mask = point_cloud[:,4].to(int) == frame_id
        frame = point_cloud[frame_mask,:].to(device)

        frame_valid_mask,refined_valid_mask = pipeline(frame,13)
        final_mask[frame_mask] = frame_valid_mask.to(bool).to(device)
        final_mask[frame_mask] = refined_valid_mask
        #TODO fix refinement

    return final_mask,final_mask_refined


def segmentation_main(data_dict):

    point_cloud = torch.from_numpy(data_dict['data']).float()

    min_frame = point_cloud[:,4].min().item()
    max_frame = point_cloud[:,4].max().item()

    #point_cloud = point_cloud.to(device)
    final_mask,_ = pipeline_frames(point_cloud,int(min_frame),5)#int(max_frame))
    #temporary fix of intensity tresholding
    final_mask = final_mask.cpu()
    final_mask[point_cloud[:,3] <= 110] = 0
    visualize_points(point_cloud[point_cloud[:,4]<5].cpu().numpy(),point_cloud[final_mask,:].cpu().numpy())
    data_dict['segmentation_mask'] = final_mask
    data_dict['segmentation'] = point_cloud[final_mask].numpy()

