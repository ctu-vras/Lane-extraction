#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>

void downsamplePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, float leaf_size)
{
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(leaf_size, leaf_size, leaf_size);
  sor.filter(*cloud);
}

void statisticalOutlierRemovalFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(5);
  sor.setStddevMulThresh(5);
  sor.filter(*cloud);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr segmentGroundPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{

  pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>);

  // Normals for each point
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  ne.setSearchMethod(tree);
  ne.setKSearch(50);
  ne.compute(*cloud_normals);

  // Segment the point cloud to find the ground plane
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight(0.1);
  seg.setMaxIterations(1000);
  seg.setDistanceThreshold(0.5);
  seg.setInputCloud(cloud);
  seg.setInputNormals(cloud_normals);
  seg.segment(*inliers, *coefficients);

  // Extract the ground plane from the point cloud
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*ground);

  return ground;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr project2D(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  coefficients->values.resize(4);
  coefficients->values[0] = coefficients->values[1] = 0;
  coefficients->values[2] = 1.0;
  coefficients->values[3] = 0;

  pcl::ProjectInliers<pcl::PointXYZ> proj;
  proj.setModelType(pcl::SACMODEL_PLANE);
  proj.setInputCloud(cloud);
  proj.setModelCoefficients(coefficients);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);

  proj.filter(*cloud_projected);

  return cloud_projected;
}

int main(int argc, char *argv[])
{
  // creates a PointCloud<PointXYZI> boost shared pointer and initializes it
  pcl::PointCloud<pcl::PointXYZ>::Ptr original_cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // loads the PointCloud data
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *original_cloud) == -1)
  {
    PCL_ERROR("Couldn't read file with PointCloud data \n");
    return (-1);
  }

  pcl::PCDWriter writer;

  downsamplePointCloud(original_cloud, 0.4);
  std::cout << "Downsample generated" << std::endl;
  // writer.write<pcl::PointXYZ>("downsample.pcd", *original_cloud, false);

  pcl::PointCloud<pcl::PointXYZ>::Ptr ground = segmentGroundPlane(original_cloud);
  std::cout << "Ground plane extracted" << std::endl;
  // writer.write<pcl::PointXYZ>("plane.pcd", *ground, false);

  statisticalOutlierRemovalFilter(ground);
  std::cout << "Outliers removed" << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr projection = project2D(ground);
  std::cout << "Projection generated" << std::endl;
  writer.write<pcl::PointXYZ>("projection.pcd", *projection, false);

  return (0);
}