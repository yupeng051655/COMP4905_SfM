#include <iostream>
#include <vector>
#include <sys/uio.h>
#include <fstream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

void init_structure(
	Mat K,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<vector<DMatch>>& matches_for_all,
	vector<Point3f>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions
);

void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2
	);

void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
);
bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask);

void maskout_points(vector<Point2f>& p1, Mat& mask);

void maskout_colors(vector<Vec3b>& p1, Mat& mask);

void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure);

void get_objpoints_and_imgpoints(vector<DMatch>& matches, vector<int>& struct_indices, 
	vector<Point3f>& structure,
	vector<KeyPoint>& key_points,
	vector<Point3f>& object_points,
	vector<Point2f>& image_points);

void fusion_structure(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3f>& structure,
	vector<Point3f>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors);

void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector <vector<Vec3b>>& colors_for_all
);

void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps);

void match_features_optical_flow(vector<string>& img_names, 
								vector<vector<KeyPoint>>& key_points_for_all, 
								vector <vector<Vec3b>>&	colors_for_all,
								vector<Mat>& descriptor_for_all,
								vector<vector<DMatch>>& matches_for_all);

void match_features_ratio_test(Mat& query, Mat& train, vector<DMatch>& matches);

void match_features_crossing_match(Mat& query, Mat& train, vector<DMatch>& matches);

void match_features(vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all);

void save_structure(vector<Mat>& rotations, vector<Mat>& motions, vector<Point3f>& structure, vector<Vec3b>& colors);

void image_sort();

int minKey(double key[], bool mstSet[], int size);

void printMST(int parent[], vector<vector<double> > graph, int size);

void primMST(vector<vector<double> > graph, int size);

int main(int argc, char** argv){

	//image_sort();
	vector<string> img_names;

	ifstream fin("images.txt");
	string imageName;
	int image_count = 0;


	while (getline(fin, imageName)){
		image_count++;
		img_names.push_back(imageName);

	}
	
	//color histogram-based detection for overlapping image
	image_sort();

	//instrinsic matrix -- calcualted by Zhang's camera caliberation 
	Mat K(Matx33d(
		2560.318186152148, 0, 1158.205190011483,
		0, 2560.680386945406, 1587.586237739258,
		0, 0, 1));


	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;

	//feature point extraction
	extract_features(img_names, key_points_for_all, descriptor_for_all, colors_for_all);

	//feature point matching 
	match_features(descriptor_for_all, matches_for_all);

	//feature matching using optical flow -- dismissed
	//match_features_optical_flow(img_names, key_points_for_all, colors_for_all, descriptor_for_all, matches_for_all);

	vector<Point3f> structure;
	vector<vector<int>> correspond_struct_idx;
	vector<Vec3b> colors;
	vector<Mat> rotations;
	vector<Mat> motions;

	//initialization 3d structure based on two images
	init_structure(
		K,
		key_points_for_all,
		colors_for_all,
		matches_for_all,
		structure,
		correspond_struct_idx,
		colors,
		rotations,
		motions
	);

	//iterative reconstruction, add images using incremental structure of motion
	for (int i = 1; i < matches_for_all.size(); ++i)
	{
		vector<Point3f> object_points;
		vector<Point2f> image_points;
		Mat r, R, T;

		get_objpoints_and_imgpoints(
			matches_for_all[i],
			correspond_struct_idx[i],
			structure,
			key_points_for_all[i + 1],
			object_points,
			image_points
		);

		//solve the translation matrix
		solvePnPRansac(object_points, image_points, K, noArray(), r, T);
		// change rotation vector to rotation matrix
		Rodrigues(r, R);
		rotations.push_back(R);
		motions.push_back(T);

		vector<Point2f> p1, p2;
		vector<Vec3b> c1, c2;
		get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i], p1, p2);
		get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i], c1, c2);

		// process triangulation based on rotation and translation vector
		vector<Point3f> next_structure;
		reconstruct(K, rotations[i], motions[i], R, T, p1, p2, next_structure);

		fusion_structure(
			matches_for_all[i],
			correspond_struct_idx[i],
			correspond_struct_idx[i + 1],
			structure,
			next_structure,
			colors,
			c1
		);
	}

	//save result
	save_structure(rotations, motions, structure, colors);
	cout << "successful!!!" << endl;
	getchar();
	
	
	return 0;

}

// A utility function to find the vertex with  
// minimum key value, from the set of vertices  
// not yet included in MST  
int minKey(double key[], bool mstSet[], int size){

    // Initialize min value  
    //double min = INT_MAX, min_index;
    double min = 9999, min_index; 
  
    for (int v = 0; v < size; v++)  
        if (mstSet[v] == false && key[v] < min)  
            min = key[v], min_index = v;  
  
    return min_index;  
}  
  
// A utility function to print the  
// constructed MST stored in parent[]  
void printMST(int parent[], vector<vector<double> > graph, int size){

    cout<<"Edge \tWeight\n";  
    for (int i = 1; i < size; i++)  
        cout<<parent[i]<<" - "<<i<<" \t"<<graph[i][parent[i]]<<" \n";

    vector<double> res;

    for (int i = 1; i < graph.size(); i++){
    	for(int j = 1; j < res.size(); j++){
    		if(res[j] == parent[i] || res[j] == i){
    			res[j] = 0;
    		}
    	}
    	res.push_back(parent[i]);
    	res.push_back(i);
    }

    vector<int> new_res;

    for (int i = 1; i < graph.size(); i++){

    int add_parent = 0;
    int add_child = 0;
    int add_middle = 0;
    int position = 0;

    	for(int j = 0; j < new_res.size(); j++){

    		if(new_res[j] == parent[i]){
    			add_parent = 1;
    		}
    	}

    	if(add_parent == 0){
    		for(int k = 0; k < new_res.size(); k++){
    			if(new_res[k] == i){
    				add_middle = 1;
    				position = k;
    				break;
    			}

    		}
    		if(add_middle == 1){
    			new_res.insert(new_res.begin()+position, parent[i]);
    		}
    		else{
    			new_res.push_back(parent[i]);
    		}	
    	}

    	for(int j = 0; j < new_res.size(); j++){
    		if(new_res[j] == i){
    			add_child = 1;
    		}
    	}
    	if(add_child == 0){
    			new_res.push_back(i);
    	}
    }

    ofstream fs("new_order.txt");
    for(int i = 0; i < new_res.size(); i++){

        if(i == 0){
            fs << "./dog/00" << new_res[i]+1 << ".jpg\n";
        }
        else{
            if(new_res[i] != 0){
                fs << "./dog/00" << new_res[i]+1 << ".jpg";
                if(i < new_res.size()-1){
	        		fs << "\n";
	        	}
            }         
        }
    }
    cout << endl;
   
}

  
// Function to construct and print MST for  
// a graph represented using adjacency  
// matrix representation  
void primMST(vector<vector<double> > graph, int size){  
    // Array to store constructed MST  
    int parent[size];  
      
    // Key values used to pick minimum weight edge in cut  
    double key[size];  
      
    // To represent set of vertices included in MST  
    bool mstSet[size];  
  
    // Initialize all keys as INFINITE  
    for (int i = 0; i < size; i++)  
        //key[i] = INT_MAX, mstSet[i] = false;
        key[i] = 9999, mstSet[i] = false;  
  
    // Always include first 1st vertex in MST.  
    // Make key 0 so that this vertex is picked as first vertex.  
    key[0] = 0;  
    parent[0] = -1; // First node is always root of MST  
  
    // The MST will have V vertices  
    for (int count = 0; count < size - 1; count++) 
    {  
        // Pick the minimum key vertex from the  
        // set of vertices not yet included in MST  
        int u = minKey(key, mstSet, size);  
  
        // Add the picked vertex to the MST Set  
        mstSet[u] = true;  
  
        // Update key value and parent index of  
        // the adjacent vertices of the picked vertex.  
        // Consider only those vertices which are not  
        // yet included in MST  
        for (int v = 0; v < size; v++)  
  
            // graph[u][v] is non zero only for adjacent vertices of m  
            // mstSet[v] is false for vertices not yet included in MST  
            // Update the key only if graph[u][v] is smaller than key[v]  
            if (graph[u][v] && mstSet[v] == false && graph[u][v] < key[v])  
                parent[v] = u, key[v] = graph[u][v];  
    }  
  
    // print the constructed MST  
    printMST(parent, graph, size);  
}

void image_sort(){

	ifstream fin("images_List_3.txt");

	vector<Mat> images;
	string imageName;
	int image_count = 0;

	while (getline(fin, imageName)){
		image_count++;
		Mat tmp_img = imread(imageName);
		cout << imageName << endl;
		images.push_back(tmp_img);
	}

	cout << "image count: " << image_count << endl;
	/// Convert to HSV
    vector<Mat> hsv_list;

    for(int i = 0; i < images.size(); i++){
        Mat tmp_hsv;
        cvtColor(images[i], tmp_hsv, COLOR_BGR2HSV);
        hsv_list.push_back(tmp_hsv);
    }

    /// Using 50 bins for hue and 60 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = {50, 60};

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };

    const float* ranges[] = { h_ranges, s_ranges };

    // Use the o-th and 1-st channels
    int channels[] = { 0, 1 };

    /// Calculate the histograms for the HSV images
    vector<MatND> hist_list;
    for(int i=0; i<hsv_list.size(); i++){
        MatND tmp_hist;
        calcHist(&hsv_list[i], 1, channels, Mat(), tmp_hist, 2, histSize, ranges, true, false);
        normalize(tmp_hist, tmp_hist, 0, 1, NORM_MINMAX, -1, Mat());
        hist_list.push_back(tmp_hist);
    }

    vector<vector<double> > res_vector;
    //double res[V][V];
    for(int i=0; i<hsv_list.size(); i++){

        vector<double> tmp_res;
        for(int j=0; j<hsv_list.size(); j++){

            double result = compareHist(hist_list[i], hist_list[j], 0); //0 means using correlation
            tmp_res.push_back(1-result); 
        }
        res_vector.push_back(tmp_res);
    }

    primMST(res_vector, res_vector.size());

}


void save_structure(vector<Mat>& rotations, vector<Mat>& motions, vector<Point3f>& structure, vector<Vec3b>& colors){
	ofstream fs("structure.ply");
	fs << "ply\n" << "format ascii 1.0\n" << "element face 0\n";
	fs << "property list uchar int vertex_indices\n";
	fs << "element vertex " << structure.size() << "\n";
	fs << "property float x\n";
	fs << "property float y\n";
	fs << "property float z\n"; 
	fs << "property uchar diffuse_red\n";
	fs << "property uchar diffuse_green\n";
	fs << "property uchar diffuse_blue\n";
	fs << "end_header\n";

	for (size_t i = 0; i < structure.size(); ++i)
	{
		fs << structure[i].x << " " << structure[i].y << " " << structure[i].z << " ";
		int r = colors[i].val[0];
		int g = colors[i].val[1];
		int b = colors[i].val[2];
		fs << r << " " << g << " "<< b << "\n";	
	}
}


void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector <vector<Vec3b>>& colors_for_all){

	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat image;

	// read image and obtain the feature points
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
	for (auto it = image_names.begin(); it != image_names.end(); ++it)
	{
		image = imread(*it);
		if (image.empty())
		{
			continue;
		}
		cout << "Extracting features: " << *it << endl;

		vector<KeyPoint> key_points;
		Mat descriptor;
		// Detects keypoints and computes the descriptors

		cv::resize(image, image, cv::Size(), 0.8, 0.8);
		sift->detect(image, key_points);
		sift->compute(image, key_points, descriptor);

		// ignore this image if no enough feature point
		if (key_points.size() <= 10)
		{
			continue;
		}

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		vector<Vec3b> colors(key_points.size());	// store the point
		for (int i = 0; i < key_points.size(); ++i)
		{
			Point2f& p = key_points[i].pt;
			if (p.x <= image.rows && p.y <= image.cols)
				colors[i] = image.at<Vec3b>(p.x, p.y);
		}
		colors_for_all.push_back(colors);
	}
}

void match_features_optical_flow(vector<string>& image_names, 
								vector<vector<KeyPoint>>& key_points_for_all, 
								vector <vector<Vec3b>>&	colors_for_all,
								vector<Mat>& descriptor_for_all,
								vector<vector<DMatch>>& matches_for_all){

	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat image1;
	Mat image2;

	for (int i = 0; i < image_names.size()-1; ++i){

		image1 = imread(image_names[i]);
		image2 = imread(image_names[i+1]);


		cout << "Extracting features: " << image_names[i] << endl;

		cv::resize(image1, image1, cv::Size(), 0.8, 0.8);
		cv::resize(image2, image2, cv::Size(), 0.8, 0.8);

		vector<KeyPoint>left_keypoints,right_keypoints;

		Ptr<FastFeatureDetector> ffd=FastFeatureDetector::create();
		ffd->detect(image1, left_keypoints);
		ffd->detect(image2, right_keypoints);

		key_points_for_all.push_back(left_keypoints);
		vector<Vec3b> colors(left_keypoints.size()); // store the RBG
		for (int i = 0; i < left_keypoints.size(); ++i)
		{
			Point2f& p = left_keypoints[i].pt;
			if (p.x <= image1.rows && p.y <= image1.cols)
				colors[i] = image1.at<Vec3b>(p.x, p.y);
		}
		colors_for_all.push_back(colors);

		if(i == image_names.size()-2){
			key_points_for_all.push_back(right_keypoints);
			vector<Vec3b> colors(right_keypoints.size());
			for (int i = 0; i < right_keypoints.size(); ++i)
			{
				Point2f& p = right_keypoints[i].pt;
				if (p.x <= image2.rows && p.y <= image2.cols)
					colors[i] = image2.at<Vec3b>(p.x, p.y);
			}
			colors_for_all.push_back(colors);
		}


		vector<Point2f>left_points;
		KeyPointsToPoints(left_keypoints,left_points);

		vector<Point2f>right_points(left_points.size());
		KeyPointsToPoints(right_keypoints, right_points);

		// making sure images are grayscale
		Mat prevgray,gray;

		if (image1.channels() == 3){
			cvtColor(image1,prevgray,CV_RGB2GRAY);
			cvtColor(image2,gray,CV_RGB2GRAY);
		}
		else{
			prevgray = image1;
			gray = image2;
		}

		// Calculate the optical flow field:
		// how each left_point moved across the 2 images
		vector<uchar>vstatus; 
		vector<float>verror;
		calcOpticalFlowPyrLK(prevgray, gray, left_points, right_points, vstatus, verror);

		// First, filter out the points with high error
		vector<Point2f>right_points_to_find;
		vector<int>right_points_to_find_back_index;

		for (unsigned int i=0; i<vstatus.size(); i++){
			if (vstatus[i] && verror[i] < 12.0){
				// Keep the original index of the point in the
				// optical flow array, for future use
				right_points_to_find_back_index.push_back(i);
				// Keep the feature point itself
				right_points_to_find.push_back(right_points[i]);
			}
			else{
				vstatus[i] = 0; // a bad flow
			}
		}

		// for each right_point see which detected feature it belongs to
		Mat right_points_to_find_flat = Mat(right_points_to_find).reshape(1,right_points_to_find.size()); //flatten array
		vector<Point2f>right_features; // detected features
		KeyPointsToPoints(right_keypoints,right_features);
		Mat right_features_flat = Mat(right_features).reshape(1,right_features.size());

		// Look around each OF point in the right image
		// for any features that were detected in its area
		// and make a match.

		BFMatcher matcher(CV_L2);
		vector<vector<DMatch>> nearest_neighbors;
		matcher.radiusMatch(right_points_to_find_flat, right_features_flat, nearest_neighbors, 2.0f);

		// Check that the found neighbors are unique (throw away neighbors
		// that are too close together, as they may be confusing)

		vector<DMatch> matches;
		std::set<int>found_in_right_points; // for duplicate prevention
		for(int i=0;i<nearest_neighbors.size();i++){
			DMatch _m;
			if(nearest_neighbors[i].size()==1){
				_m = nearest_neighbors[i][0]; // only one neighbor
			}
			else if(nearest_neighbors[i].size()>1){
				//ratio test 
				// 2 neighbors – check how close they are
				double ratio = nearest_neighbors[i][0].distance / nearest_neighbors[i][1].distance;
				if(ratio < 0.7){
					_m = nearest_neighbors[i][0];
				}
				else{
					continue;
				}
			}
			else{
				continue; // no neighbors... :(
			}

			// prevent duplicates
			if (found_in_right_points.find(_m.trainIdx) == found_in_right_points.end()){
				// The found neighbor was not yet used:
				// We should match it with the original indexing 
				// ofthe left point
				_m.queryIdx = right_points_to_find_back_index[_m.queryIdx]; 
				matches.push_back(_m);
				
			}

		}
		matches_for_all.push_back(matches);
	}

}


void match_features(vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all){
	matches_for_all.clear();

	//mathcing with its neighbouring image
	for (int i = 0; i < descriptor_for_all.size() - 1; ++i)
	{
		cout << "Matching images " << i << " - " << i + 1 << endl;
		vector<DMatch> matches;
		match_features_ratio_test(descriptor_for_all[i], descriptor_for_all[i + 1], matches);
		matches_for_all.push_back(matches);
	}
}


void match_features_ratio_test(Mat& query, Mat& train, vector<DMatch>& matches){
	vector<vector<DMatch>> knn_matches;

	BFMatcher matcher(NORM_L2);

	const float minRatio = 1.f / 1.5f;
	const int k = 2;

	matcher.knnMatch(query, train, knn_matches, k);

	//using ratio test obtain the minimum distance in matching
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		// Rotio Test
		if (knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance)
		{
			continue;
		}

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist)
		{
			min_dist = dist;
		}
	}

	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//ratio test
		if (
			knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
		{
			continue;
		}

		//save good matchers
		matches.push_back(knn_matches[r][0]);
	}
	
}


void init_structure(
	Mat K,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<vector<DMatch>>& matches_for_all,
	vector<Point3f>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions){

	//calculate the translation matrix of first two image
	vector<Point2f> p1, p2;
	vector<Vec3b> c2;
	Mat R, T;	//rotation and translation vector 
	Mat mask;	//In mask, if value > 0, it represents good matching point. If value = 0, it represents outlier.
	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
	get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);
	find_transform(K, p1, p2, R, T, mask);	//  decompose essential matrix get rotation and translation vector 

	// 3d reconstruction 
	maskout_points(p1, mask);
	maskout_points(p2, mask);
	maskout_colors(colors, mask);

	Mat R0 = Mat::eye(3, 3, CV_64FC1);
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);

	//triangulation
	reconstruct(K, R0, T0, R, T, p1, p2, structure);	

	rotations = { R0, R };
	motions = { T0, T };


	correspond_struct_idx.clear();
	correspond_struct_idx.resize(key_points_for_all.size());
	for (int i = 0; i < key_points_for_all.size(); ++i)
	{
		correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);
	}

	int idx = 0;
	vector<DMatch>& matches = matches_for_all[0];
	for (int i = 0; i < matches.size(); ++i)
	{
		if (mask.at<uchar>(i) == 0)
		{
			continue;
		}

		// if the idx of the two points are equal, it indicates that they are same feature point
		correspond_struct_idx[0][matches[i].queryIdx] = idx;	
		correspond_struct_idx[1][matches[i].trainIdx] = idx;
		++idx;
	}
}

void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2){

	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}

void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2){

	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask){

	// obtain focal length and center point from intrinsic matrix 
	double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	// find the essential matrix and use RANSAC to filter the matching points
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty()){
		return false;
	}

	double feasible_count = countNonZero(mask);	// good matching 

	// threshold > 0.6 
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6){
		cout << "feasible matches are lower than 60%" << endl;
		return false;
	}
	
	// decompose the essential matrix
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	if (((double)pass_count) / feasible_count < 0.8){
		cout << "pass matches are lower than 80^%" << endl;
		return false;
	}
	return true;
}

void maskout_points(vector<Point2f>& p1, Mat& mask){
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
		{
			p1.push_back(p1_copy[i]);
		}
	}
}

void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
		{
			p1.push_back(p1_copy[i]);
		}
	}
}

void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure){

	//projection matrix[R, T]
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
	T1.convertTo(proj1.col(3), CV_32FC1);

	R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T2.convertTo(proj2.col(3), CV_32FC1);

	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK * proj1;
	proj2 = fK * proj2;

	// triangulation
	Mat s;
	triangulatePoints(proj1, proj2, p1, p2, s);

	structure.clear();
	structure.reserve(s.cols);
	for (int i = 0; i < s.cols; ++i)
	{
		Mat_<float> col = s.col(i);
		col /= col(3);	//homogeneous coordinate 
		structure.push_back(Point3f(col(0), col(1), col(2)));
	}
}

void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<Point3f>& structure,
	vector<KeyPoint>& key_points,
	vector<Point3f>& object_points,
	vector<Point2f>& image_points){

	object_points.clear();
	image_points.clear();

	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx < 0)	
		{
			continue;
		}

		object_points.push_back(structure[struct_idx]);
		image_points.push_back(key_points[train_idx].pt);	
	}
}

void fusion_structure(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3f>& structure,
	vector<Point3f>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors){

	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx >= 0)
		{
			next_struct_indices[train_idx] = struct_idx;
			continue;
		}

		structure.push_back(next_structure[i]);
		colors.push_back(next_colors[i]);
		struct_indices[query_idx] = next_struct_indices[train_idx] = structure.size() - 1;
	}
}

void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps){
	ps.clear();
	for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}



