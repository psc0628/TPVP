#include <windows.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
typedef unsigned long long pop_t;

using namespace std;

#include "Share_Data.hpp"
#include "View_Space.hpp"
#include "Information.hpp"

//Virtual_Perception_3D.hpp
void precept_thread_process(int i, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, octomap::point3d* _origin, octomap::point3d* _end, Eigen::Matrix4d* _view_pose_world, octomap::ColorOcTree* _ground_truth_model, Share_Data* share_data);

class Perception_3D {
public:
	Share_Data* share_data;
	octomap::ColorOcTree* ground_truth_model;
	int full_voxels;

	Perception_3D(Share_Data* _share_data) {
		share_data = _share_data;
		ground_truth_model = new octomap::ColorOcTree(share_data->ground_truth_resolution);
		for (octomap::ColorOcTree::leaf_iterator it = share_data->ground_truth_model->begin_leafs(), end = share_data->ground_truth_model->end_leafs(); it != end; ++it) {
			ground_truth_model->setNodeValue(it.getX(), it.getY(), it.getZ(), it->getLogOdds(), true);
			ground_truth_model->setNodeColor(it.getX(), it.getY(), it.getZ(), 255, 0, 0);
		}
		if (share_data->has_table) {
			for (double x = share_data->object_center_world(0) - 0.2; x <= share_data->object_center_world(0) + 0.2; x += share_data->ground_truth_resolution)
				for (double y = share_data->object_center_world(1) - 0.2; y <= share_data->object_center_world(1) + 0.2; y += share_data->ground_truth_resolution) {
					double z = share_data->min_z_table;
					ground_truth_model->setNodeValue(x, y, z, ground_truth_model->getProbHitLog(), true);
					ground_truth_model->setNodeColor(x, y, z, 0, 0, 255);
				}
			//ground_truth_model->write(share_data->save_path + "/GT_table.ot");
		}
		full_voxels = 0;
		for (octomap::ColorOcTree::leaf_iterator it = ground_truth_model->begin_leafs(), end = ground_truth_model->end_leafs(); it != end; ++it) {
			full_voxels++;
		}
	}

	~Perception_3D() {
		delete ground_truth_model;
	}

	bool precept(View* now_best_view) {
		//���ʹ�ñ���ĵ��Ƽ��ٵĻ������Զ�ȡ
		if (share_data->use_saved_cloud) {
			int view_id = now_best_view->id;
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr no_table(new pcl::PointCloud<pcl::PointXYZRGB>);
			string view_cloud_file_path = share_data->gt_path + "/GT_points/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "/cloud_view" + to_string(view_id) + ".pcd";
			string view_cloud_notable_file_path = share_data->gt_path + "/GT_points/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "/cloud_notable_view" + to_string(view_id) + ".pcd";
			if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(view_cloud_file_path, *cloud) != -1 && pcl::io::loadPCDFile<pcl::PointXYZRGB>(view_cloud_notable_file_path, *no_table) != -1) {
				cout << "Load view clouds success. Use saved cloud to speed up evaluation." << endl;
				//��¼��ǰ�ɼ�����
				share_data->vaild_clouds++;
				share_data->clouds.push_back(cloud);
				//��ת����������ϵ
				share_data->clouds_notable.push_back(no_table);
				*share_data->cloud_final += *no_table;
				return true;
			}
			else {
				cout << "Load view cloud failed. Use virtual perception." << endl;
			}
		}
		//�����ʹ�ñ������ݻ��ȡʧ�ܣ���������
		double now_time = clock();
		//������ǰ�������
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_parallel(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_parallel->is_dense = false;
		cloud_parallel->points.resize(full_voxels);
		//��ȡ�ӵ�λ��
		Eigen::Matrix4d view_pose_world;
		now_best_view->get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
		view_pose_world = (share_data->now_camera_pose_world * now_best_view->pose.inverse()).eval();
		//����ӵ��key
		octomap::OcTreeKey key_origin;
		bool key_origin_have = ground_truth_model->coordToKeyChecked(now_best_view->init_pos(0), now_best_view->init_pos(1), now_best_view->init_pos(2), key_origin);
		if (key_origin_have) {
			octomap::point3d origin = ground_truth_model->keyToCoord(key_origin);
			//����ÿ������
			octomap::point3d* end = new octomap::point3d[full_voxels];
			octomap::ColorOcTree::leaf_iterator it = ground_truth_model->begin_leafs();
			for (int i = 0; i < full_voxels; i++) {
				end[i] = it.getCoordinate();
				it++;
			}
			//ground_truth_model->write(share_data->save_path + "/test_camrea.ot");
			//���̴߳���
			vector<thread> precept_process;
			for (int i = 0; i < full_voxels; i += share_data->max_num_of_thread) {
				for (int j = 0; j < share_data->max_num_of_thread && i + j < full_voxels; j++) {
					precept_process.push_back(thread(precept_thread_process, i + j, cloud_parallel, &origin, &end[i + j], &view_pose_world, ground_truth_model, share_data));
				}
				for (int j = 0; j < share_data->max_num_of_thread && i + j < full_voxels; j++) {
					precept_process[i + j].join();
				}
			}
			//�ͷ��ڴ�
			delete[] end;
			precept_process.clear();
			precept_process.shrink_to_fit();
		}
		else {
			cout << "View out of map.check." << endl;
		}
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr no_table(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud->is_dense = false;
		no_table->is_dense = false;
		cloud->points.resize(full_voxels);
		no_table->points.resize(full_voxels);
		auto ptr = cloud->points.begin();
		auto pt = no_table->points.begin();
		int vaild_point = 0;
		int table_point = 0;
		auto p = cloud_parallel->points.begin();
		for (int i = 0; i < cloud_parallel->points.size(); i++, p++)
		{
			if ((*p).x == 0 && (*p).y == 0 && (*p).z == 0) continue;
			if ((*p).z > share_data->min_z_table + share_data->ground_truth_resolution) {
				(*pt).x = (*p).x;
				(*pt).y = (*p).y;
				(*pt).z = (*p).z;
				(*pt).b = (*p).b;
				(*pt).g = (*p).g;
				(*pt).r = (*p).r;
				table_point++;
				pt++;
			}
			(*ptr).x = (*p).x;
			(*ptr).y = (*p).y;
			(*ptr).z = (*p).z;
			(*ptr).b = (*p).b;
			(*ptr).g = (*p).g;
			(*ptr).r = (*p).r;
			vaild_point++;
			ptr++;
		}
		cloud->width = vaild_point;
		no_table->width = table_point;
		cloud->height = 1;
		no_table->height = 1;
		cloud->points.resize(vaild_point);
		no_table->points.resize(table_point);
		//��¼��ǰ�ɼ�����
		share_data->vaild_clouds++;
		share_data->clouds.push_back(cloud);
		//��ת����������ϵ
		share_data->clouds_notable.push_back(no_table);
		*share_data->cloud_final += *no_table;
		//cout << "virtual cloud num is " << vaild_point << endl;
		//cout << "virtual cloud table num is " << table_point << endl;
		//cout << "virtual cloud get with executed time " << clock() - now_time << " ms." << endl;
		if (share_data->show) { //��ʾ�������
			pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Camera"));
			viewer1->setBackgroundColor(255, 255, 255);
			//viewer1->addCoordinateSystem(0.1);
			viewer1->initCameraParameters();
			viewer1->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
			Eigen::Vector4d X(0.05, 0, 0, 1);
			Eigen::Vector4d Y(0, 0.05, 0, 1);
			Eigen::Vector4d Z(0, 0, 0.05, 1);
			Eigen::Vector4d O(0, 0, 0, 1);
			X = view_pose_world * X;
			Y = view_pose_world * Y;
			Z = view_pose_world * Z;
			O = view_pose_world * O;
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
			
			/* // show rays
			//viewer1->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_gt");
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(-1));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(-1));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(-1));
			for (int i = 0; i < cloud->points.size(); i+=1000) {
				viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z), 0.75, 0.75, 0, "point" + to_string(i));
			}
			viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
			*/
			while (!viewer1->wasStopped())
			{
				viewer1->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
		}

		cloud_parallel->points.clear();
		cloud_parallel->points.shrink_to_fit();

		cout <<"Virtual cloud getted with time "<< clock() - now_time<<" ms." << endl;
		return true;
	}
};

void precept_thread_process(int i, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, octomap::point3d* _origin, octomap::point3d* _end, Eigen::Matrix4d* _view_pose_world, octomap::ColorOcTree* _ground_truth_model, Share_Data* share_data) {
	//num++;
	octomap::point3d origin = *_origin;
	Eigen::Matrix4d view_pose_world = *_view_pose_world;
	octomap::ColorOcTree* ground_truth_model = _ground_truth_model;
	pcl::PointXYZRGB point;
	point.x = 0; point.y = 0; point.z = 0;
	//ͶӰ����Ƿ��ڳ���Χ��
	Eigen::Vector4d end_3d(_end->x(), _end->y(), _end->z(), 1);
	Eigen::Vector4d vertex = view_pose_world.inverse() * end_3d;
	float point_3d[3] = { vertex(0), vertex(1),vertex(2) };
	float pixel[2];
	rs2_project_point_to_pixel(pixel, &share_data->color_intrinsics, point_3d);
	if (pixel[0] < 0 || pixel[0]>share_data->color_intrinsics.width || pixel[1] < 0 || pixel[1]>share_data->color_intrinsics.height) {
		cloud->points[i] = point;
		return;
	}
	//����ͶӰ�ҵ��յ�
	octomap::point3d end = project_pixel_to_ray_end(pixel[0], pixel[1], share_data->color_intrinsics, view_pose_world, 1.0);
	octomap::OcTreeKey key_end;
	octomap::point3d direction = end - origin;
	octomap::point3d end_point;
	//Խ��δ֪�����ҵ��յ�
	bool found_end_point = ground_truth_model->castRay(origin, direction, end_point, true, 1.0);
	if (!found_end_point) {//δ�ҵ��յ㣬�޹۲�����
		cloud->points[i] = point;
		return;
	}
	if (end_point == origin) {
		cout << "view in the object. check!" << endl;
		cloud->points[i] = point;
		return;
	}
	//���һ��ĩ���Ƿ��ڵ�ͼ���Ʒ�Χ��
	bool key_end_have = ground_truth_model->coordToKeyChecked(end_point, key_end);
	if (key_end_have) {
		octomap::ColorOcTreeNode* node = ground_truth_model->search(key_end);
		if (node != NULL) {
			octomap::ColorOcTreeNode::Color color = node->getColor();
			point.x = end_point.x();
			point.y = end_point.y();
			point.z = end_point.z();
			point.b = color.b;
			point.g = color.g;
			point.r = color.r;
		}
	}
	cloud->points[i] = point;
}

//Global_Path_Planner.hpp
class Global_Path_Planner {
public:
	Share_Data* share_data;
	View_Space* view_space;
	View* now_view;
	bool solved;
	long long n,m;
	map<int, int>* view_id_in;
	map<int, int>* view_id_out;
	vector<vector<double>> graph;
	vector<vector<double>> dp;
	vector<vector<int>> path;
	double total_shortest;
	vector<int> global_path;

	Global_Path_Planner(Share_Data* _share_data, View_Space* _view_space, View* _now_view, vector<int>& view_set_label) {
		share_data = _share_data;
		view_space = _view_space;
		now_view = _now_view;
		solved = false;
		//�����±�ӳ��
		view_id_in = new map<int, int>();
		view_id_out = new map<int, int>();
		(*view_id_in)[now_view->id] = 0;
		(*view_id_out)[0] = now_view->id;
		for (int i = 0; i < view_set_label.size(); i++) {
			(*view_id_in)[view_set_label[i]] = i+1;
			(*view_id_out)[i+1] = view_set_label[i];
		}
		//�ڵ����뷽����
		n = view_set_label.size() + 1;
		m = 1LL << n;
		//local path ��ȫ����ͼ
		graph.resize(n);
		for (int i = 0; i < n; i++)
			graph[i].resize(n);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				//����id
				int u = (*view_id_out)[i];
				int v = (*view_id_out)[j];
				//������·��
				pair<int, double> local_path = get_local_path(view_space->views[u].init_pos.eval(), view_space->views[v].init_pos.eval(), view_space->object_center_world.eval(), view_space->predicted_size * sqrt(2)); //��Χ�а뾶�ǰ�߳��ĸ���2��
				if (local_path.first < 0) cout << "local path wrong." << endl;
				graph[i][j] = local_path.second;
			}
		//��ʼ��dp
		dp.resize(m);
		for (int i = 0; i < m; i++)
			dp[i].resize(n);
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				dp[i][j] = 1e10;
		dp[1][0] = 0;
		//��ʼ��·����¼
		path.resize(m);
		for (int i = 0; i < m; i++)
			path[i].resize(n);
		cout << "Global_Path_Planner inited." << endl;
	}

	~Global_Path_Planner() {
		view_id_in->clear();
		delete view_id_in;
		view_id_out->clear();
		delete view_id_out;
		for (int i = 0; i < graph.size(); i++) {
			graph[i].clear();
			graph[i].shrink_to_fit();
		}
		graph.clear();
		graph.shrink_to_fit();
		for (int i = 0; i < dp.size(); i++) {
			dp[i].clear();
			dp[i].shrink_to_fit();
		}
		dp.clear();
		dp.shrink_to_fit();
		for (int i = 0; i < path.size(); i++) {
			path[i].clear();
			path[i].shrink_to_fit();
		}
		path.clear();
		path.shrink_to_fit();
		global_path.clear();
		global_path.shrink_to_fit();
	}

	double solve() {
		double now_time = clock();
		for (int i = 0; i < m; i++)  // i�������һ�������ļ��ϣ�����ÿ��λ�õ�0/1����û��/�о��������(m=1<<n)
		{
			for (int j = 0; j < n; j++)  //ö�ٵ�ǰ�����ĸ���
			{
				if ((i >> j) & 1)  //���i�����j
				{
					for (int k = 0; k < n; k++)  //ö�ٵ���j�ĵ�
					{
						if (i - (1 << j) >> k & 1)  //ȥ������j�ļ���i
						{
							//�����������²���¼ת��·��
							if (dp[i][j] >= dp[i - (1LL << j)][k] + graph[k][j]) {
								dp[i][j] = dp[i - (1LL << j)][k] + graph[k][j];
								path[i][j] = k;
							}
							// dp[i][j] = min(dp[i][j], dp[i - (1 << j)][k] + mp[k][j]);
						}
					}
				}
			}
		}
		//Ĭ�����Ϊ0,����ÿ�������յ��ҵ���̹��ܶ���·����ȫ���Ƿ�����
		int end_node;
		total_shortest = 1e20;
		for (int i = 1; i < n; i++) {
			if (total_shortest > dp[m - 1][i]) {
				total_shortest = dp[m - 1][i];
				end_node = i;
			}
		}
		//���·��
		for (int i = (1 << n) - 1, j = end_node; i > 0;) {
			//ע���±꽻��
			global_path.push_back((*view_id_out)[j]);
			int ii = i - (1 << j);
			int jj = path[i][j];
			i = ii, j = jj;
		}
		//����·���跴��
		reverse(global_path.begin(), global_path.end());
		solved = true;
		double cost_time = clock() - now_time;
		cout << "Global Path length " << total_shortest << " getted with executed time " << cost_time << " ms." << endl;
		//����
		share_data->access_directory(share_data->save_path + "/movement");
		ofstream fout_global_path(share_data->save_path + "/movement/global_path.txt");
		fout_global_path << total_shortest << '\t' << cost_time << '\t' << endl;
		return total_shortest;
	}

	vector<int> get_path_id_set() {
		if (!solved) cout << "call solve() first" << endl;
		cout << "Node ids on global_path form start to end are: ";
		for (int i = 0; i < global_path.size(); i++)
			cout << global_path[i] << " ";
		cout << endl;
		//ɾ��������
		vector<int> ans;
		ans = global_path;
		ans.erase(ans.begin());
		return ans;
	}
};

//views_voxels_LM.hpp
//����GT�ļ��ϸ���
class views_voxels_LM {
public:
	Share_Data* share_data;
	View_Space* view_space;
	vector<vector<bool>> graph;
	unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_id_map;	//�����±�
	int num_of_voxel;
	set<int> chosen_views;
	GRBEnv* env;
	GRBModel* model;
	vector<GRBVar> x;
	GRBLinExpr obj;

	void solve() {
		// Optimize model
		model->optimize();
		// show nonzero variables
		/*for (int i = 0; i < share_data->num_of_views; i++)
			if (x[i].get(GRB_DoubleAttr_X) == 1.0)
				cout << x[i].get(GRB_StringAttr_VarName) << " " << x[i].get(GRB_DoubleAttr_X) << endl;
		// show num of views
		cout << "Obj: " << model->get(GRB_DoubleAttr_ObjVal) << endl;*/
	}

	vector<int> get_view_id_set() {
		vector<int> ans;
		for (int i = 0; i < share_data->num_of_views; i++)
			if (x[i].get(GRB_DoubleAttr_X) == 1.0) ans.push_back(i);
		return ans;
	}

	views_voxels_LM(Share_Data* _share_data, View_Space* _view_space, set<int> _chosen_views, vector<unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>*> voxels) {
		double now_time = clock();
		share_data = _share_data;
		view_space = _view_space;
		chosen_views = _chosen_views;
		//�������ص�id��
		num_of_voxel = 0;
		voxel_id_map = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		for (int i = 0; i < voxels.size(); i++) {
			for (auto& it : *voxels[i]) {
				if (voxel_id_map->find(it.first) == voxel_id_map->end()) {
					(*voxel_id_map)[it.first] = num_of_voxel++;
				}
			}
		}
		//cout << num_of_voxel << " real | gt " << share_data->full_voxels << endl;
		graph.resize(num_of_voxel);
		for (int i = 0; i < share_data->num_of_views; i++) {
			graph[i].resize(num_of_voxel);
			for (int j = 0; j < num_of_voxel; j++) {
				graph[i][j] = 0;
			}
		}
		set<int> voxels_not_need;
		for (int i = 0; i < voxels.size(); i++) {
			for (auto& it : *voxels[i]) {
				graph[i][(*voxel_id_map)[it.first]] = 1;
				if (chosen_views.find(i) != chosen_views.end()) {
					voxels_not_need.insert((*voxel_id_map)[it.first]);
				}
			}
		}
		//������Ӧ�����Թ滮�����
		now_time = clock();
		env = new GRBEnv();
		model = new GRBModel(*env);
		x.resize(share_data->num_of_views);
		// Create variables
		for (int i = 0; i < share_data->num_of_views; i++)
			x[i] = model->addVar(0.0, 1.0, 0.0, GRB_BINARY, "x" + to_string(i));
		// Set objective : \sum_{s\in S} x_s
		for (int i = 0; i < share_data->num_of_views; i++)
			obj += x[i];
		model->setObjective(obj, GRB_MINIMIZE);
		// Add linear constraint: \sum_{S:e\in S} x_s\geq1
		for (int j = 0; j < num_of_voxel; j++)
		{
			if (voxels_not_need.find(j) != voxels_not_need.end()) continue;
			GRBLinExpr subject_of_voxel;
			for (int i = 0; i < share_data->num_of_views; i++)
				if (graph[i][j] == 1) subject_of_voxel += x[i];
			model->addConstr(subject_of_voxel >= 1, "c" + to_string(j));
		}
		model->set("TimeLimit", "10");
		//cout << "Integer linear program formulated with executed time " << clock() - now_time << " ms." << endl;
	}

	~views_voxels_LM() {
		for (int i = 0; i < graph.size(); i++) {
			graph[i].clear();
			graph[i].shrink_to_fit();
		}
		graph.clear();
		graph.shrink_to_fit();
		chosen_views.clear();
		delete voxel_id_map;
		delete env;
		delete model;
	}
};

//NVB_Planner.hpp
#define Over 0
#define WaitData 1
#define WaitViewSpace 2
#define WaitInformation 3
#define WaitMoving 4

void save_cloud_mid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string name, Share_Data* share_data);
void create_view_space(View_Space** now_view_space, View* now_best_view, Share_Data* share_data, int iterations);
void create_views_information(Views_Information** now_views_infromation, View* now_best_view, View_Space* now_view_space, Share_Data* share_data, int iterations);
void move_robot(View* now_best_view, View_Space* now_view_space, Share_Data* share_data);
void show_cloud(pcl::visualization::PCLVisualizer::Ptr viewer);

class NBV_Planner
{
public:
	atomic<int> status;
	int iterations;
	Perception_3D* percept;
	Voxel_Information* voxel_information;
	View_Space* now_view_space;
	Views_Information* now_views_infromation;
	View* now_best_view;
	Share_Data* share_data;
	pcl::visualization::PCLVisualizer::Ptr viewer;
	bool now_views_infromation_created = false;

	~NBV_Planner() {
		delete percept;
		delete now_best_view;
		delete voxel_information;
		delete now_view_space;
		//ֻ��ʹ�ù�����������������information
		if (now_views_infromation_created) delete now_views_infromation;
	}

	double check_size(double predicted_size, Eigen::Vector3d object_center_world, vector<Eigen::Vector3d>& points) {
		int vaild_points = 0;
		for (auto& ptr : points) {
			if (ptr(0) < object_center_world(0) - predicted_size || ptr(0) > object_center_world(0) + predicted_size) continue;
			if (ptr(1) < object_center_world(1) - predicted_size || ptr(1) > object_center_world(1) + predicted_size) continue;
			if (ptr(2) < object_center_world(2) - predicted_size || ptr(2) > object_center_world(2) + predicted_size) continue;
			vaild_points++;
		}
		return (double)vaild_points / (double)points.size();
	}

	NBV_Planner(Share_Data* _share_data, int _status = WaitData) {
		share_data = _share_data;
		iterations = 0;
		share_data->iterations = 0;
		status = _status;
		share_data->now_view_space_processed = false;
		share_data->now_views_infromation_processed = false;
		share_data->move_on = false;
		voxel_information = new Voxel_Information(share_data->p_unknown_lower_bound, share_data->p_unknown_upper_bound);
		voxel_information->init_mutex_views(share_data->num_of_views);
		//ֻ�г�ʼѡ�������������Ż�������Ϣ�����࣬��ϵ�pipelineҲ���ڳ�ʼ��ʱ����������Ҫɾ��
		if (share_data->method_of_IG == 0 || share_data->method_of_IG == 1 || share_data->method_of_IG == 2 || share_data->method_of_IG == 3 || share_data->method_of_IG == 4 || share_data->method_of_IG == 5 || share_data->method_of_IG == 10) {
			now_views_infromation_created = true;
		}
		//��ʼ��GT
		//share_data->access_directory(share_data->save_path);
		//GT cloud
		share_data->cloud_ground_truth->is_dense = false;
		share_data->cloud_ground_truth->points.resize(share_data->cloud_pcd->points.size());
		share_data->cloud_ground_truth->width = share_data->cloud_pcd->points.size();
		share_data->cloud_ground_truth->height = 1;
		auto ptr = share_data->cloud_ground_truth->points.begin();
		auto p = share_data->cloud_pcd->points.begin();
		float unit = 1.0;
		for (auto& ptr : share_data->cloud_pcd->points) {
			if (fabs(ptr.x) >= 10 || fabs(ptr.y) >= 10 || fabs(ptr.z) >= 10) {
				unit = 0.001;
				cout << "change unit from <mm> to <m>." << endl;
				break;
			}
		}
		//��������С��ͳһ����Ϊ0.15m����
		vector<Eigen::Vector3d> points;
		for (auto& ptr : share_data->cloud_pcd->points) {
			Eigen::Vector3d pt(ptr.x * unit, ptr.y * unit, ptr.z * unit);
			points.push_back(pt);
		}
		Eigen::Vector3d object_center_world = Eigen::Vector3d(0, 0, 0);
		//�����������
		for (auto& ptr : points) {
			object_center_world(0) += ptr(0);
			object_center_world(1) += ptr(1);
			object_center_world(2) += ptr(2);
		}
		object_center_world(0) /= points.size();
		object_center_world(1) /= points.size();
		object_center_world(2) /= points.size();
		/*//���ֲ���BBX�뾶����BBX�ڵ�ĸ������ʴﵽ0.90-0.95Ϊ��ֹ����
		double l = 0, r = 0, mid;
		for (auto& ptr : points) {
			r = max(r, (object_center_world - ptr).norm());
		}
		mid = (l + r) / 2;
		double precent = check_size(mid, object_center_world, points);
		double pre_precent = precent;
		while (precent > 0.95 || precent < 1.0) {
			if (precent > 0.95) {
				r = mid;
			}
			else if (precent < 1.0) {
				l = mid;
			}
			mid = (l + r) / 2;
			precent = check_size(mid, object_center_world, points);
			if (fabs(pre_precent - precent) < 0.001) break;
			pre_precent = precent;
		}*/
		//������Զ��
		double predicted_size = 0.0;
		for (auto& ptr : points) {
			predicted_size = max(predicted_size, (object_center_world - ptr).norm());
		}
		predicted_size *= 17.0 / 16.0;
		//predicted_size = 0.1;

		double scale = 1.0;
		if (share_data->mp_scale.find(share_data->name_of_pcd) != share_data->mp_scale.end()) {
			scale = (predicted_size - share_data->mp_scale[share_data->name_of_pcd]) / predicted_size;
			cout << "object " << share_data->name_of_pcd << " large. change scale " << predicted_size << " to about " << predicted_size - share_data->mp_scale[share_data->name_of_pcd] << " m." << endl;
		}
		else {
			cout << "object " << share_data->name_of_pcd << " size is " << predicted_size << " m." << endl;
		}

		double predicted_octomap_resolution = -1.0;
		if (share_data->Combined_on == true || share_data->method_of_IG == 7 || share_data->method_of_IG == 6 || share_data->method_of_IG == 9) {
			//��̬�ֱ��ʣ������������
			predicted_octomap_resolution = scale * predicted_size * 2.0 / 32.0;
		}
		else {
			//�������ط���
			predicted_octomap_resolution = share_data->ground_truth_resolution * share_data->voxel_resolution_factor;
		}
		cout << "choose octomap_resolution: " << predicted_octomap_resolution << " m." << endl;
		share_data->octomap_resolution = predicted_octomap_resolution;
		share_data->octo_model = new octomap::ColorOcTree(share_data->octomap_resolution);
		share_data->octo_model->setOccupancyThres(0.65);
		share_data->GT_sample = new octomap::ColorOcTree(share_data->octomap_resolution);

		//ת������
		double min_z = object_center_world(2) * scale * unit;
		if (fabs(scale - 1.0) > 1e-6 || fabs(unit - 1.0) > 1e-6) {
			cout << "scale is " << scale << " and unit is " << unit << endl;
		}
		for (int i = 0; i < share_data->cloud_pcd->points.size(); i++, p++)
		{
			(*ptr).x = (*p).x * scale * unit;
			(*ptr).y = (*p).y * scale * unit;
			(*ptr).z = (*p).z * scale * unit;
			(*ptr).b = 0;
			(*ptr).g = 0;
			(*ptr).r = 255;
			//GT�������
			octomap::OcTreeKey key;  bool key_have = share_data->ground_truth_model->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key);
			if (key_have) {
				octomap::ColorOcTreeNode* voxel = share_data->ground_truth_model->search(key);
				if (voxel == NULL) {
					share_data->ground_truth_model->setNodeValue(key, share_data->ground_truth_model->getProbHitLog(), true);
					share_data->ground_truth_model->integrateNodeColor(key, (*ptr).r, (*ptr).g, (*ptr).b);
				}
			}
			min_z = min(min_z, (double)(*ptr).z);
			//GT_sample�������
			octomap::OcTreeKey key_sp;  bool key_have_sp = share_data->GT_sample->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key_sp);
			if (key_have_sp) {
				octomap::ColorOcTreeNode* voxel_sp = share_data->GT_sample->search(key_sp);
				if (voxel_sp == NULL) {
					share_data->GT_sample->setNodeValue(key_sp, share_data->GT_sample->getProbHitLog(), true);
					share_data->GT_sample->integrateNodeColor(key_sp, (*ptr).r, (*ptr).g, (*ptr).b);
				}
			}
			ptr++;
		}
		//cout << min_z << endl;
		//pcl::io::savePCDFile<pcl::PointXYZRGB>("C:\\Users\\yixinizhu\\Desktop\\" + share_data->name_of_pcd + ".pcd", *share_data->cloud_ground_truth);
		//��¼����
		share_data->min_z_table = min_z - share_data->ground_truth_resolution;
		cout << "min_z_table is " << share_data->min_z_table << endl;

		share_data->ground_truth_model->updateInnerOccupancy();
		//share_data->ground_truth_model->write(share_data->save_path + "/GT.ot");
		//GT_sample_voxels
		share_data->GT_sample->updateInnerOccupancy();
		//share_data->GT_sample->write(share_data->save_path + "/GT_sample.ot");
		share_data->init_voxels = 0;
		int full_voxels = 0;
		//��sample��ͳ���ܸ���
		for (octomap::ColorOcTree::leaf_iterator it = share_data->GT_sample->begin_leafs(), end = share_data->GT_sample->end_leafs(); it != end; ++it){
			if(it.getY() > share_data->min_z_table + share_data->octomap_resolution)
				share_data->init_voxels++;
			full_voxels++;
		}
		cout << "Map_GT_sample has voxels " << share_data->init_voxels << endl;
		cout << "Map_GT_sample has voxels with bottom " << full_voxels << endl;
		share_data->init_voxels = full_voxels;
		//ofstream fout_sample(share_data->save_path + "/GT_sample_voxels.txt");
		//fout_sample << share_data->init_voxels << endl;
		//��GT��ͳ���ܸ���
		share_data->cloud_points_number = 0;
		for (octomap::ColorOcTree::leaf_iterator it = share_data->ground_truth_model->begin_leafs(), end = share_data->ground_truth_model->end_leafs(); it != end; ++it) {
			share_data->cloud_points_number++;
		}
		cout << "Map_GT has voxels " << share_data->cloud_points_number << endl;
		//ofstream fout_gt(share_data->save_path + "/GT_voxels.txt");
		//fout_gt << share_data->cloud_points_number << endl;

		//��ʼ��viewspace
		int first_view_id = share_data->first_view_id;
		now_view_space = new View_Space(iterations, share_data, voxel_information, share_data->cloud_ground_truth, first_view_id);
		//���ó�ʼ�ӵ�Ϊͳһ��λ��
		//if (share_data->method_of_IG == 7) { //SC-NET
			now_view_space->views[first_view_id].vis++;
			now_best_view = new View(now_view_space->views[first_view_id]);
		//}
		/*else {
			now_view_space->views[0].vis++;
			now_best_view = new View(now_view_space->views[0]);
		}*/
		//�˶����ۣ��ӵ�id����ǰ���ۣ��������
		share_data->movement_cost = 0;
		//share_data->access_directory(share_data->save_path + "/movement");
		//ofstream fout_move(share_data->save_path + "/movement/path" + to_string(-1) + ".txt");
		//fout_move << 0 << '\t' << 0.0 << '\t' << 0.0 << endl;
		now_best_view->get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
		Eigen::Matrix4d view_pose_world = (share_data->now_camera_pose_world * now_best_view->pose.inverse()).eval();
		//������ʼ��
		percept = new Perception_3D(share_data);
		if (share_data->show) { //��ʾBBX�����λ�á�GT
			pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Iteration"));
			viewer->setBackgroundColor(0, 0, 0);
			viewer->addCoordinateSystem(0.1);
			viewer->initCameraParameters();
			//��һ֡���λ��
			Eigen::Vector4d X(0.05, 0, 0, 1);
			Eigen::Vector4d Y(0, 0.05, 0, 1);
			Eigen::Vector4d Z(0, 0, 0.05, 1);
			Eigen::Vector4d O(0, 0, 0, 1);
			X = view_pose_world * X;
			Y = view_pose_world * Y;
			Z = view_pose_world * Z;
			O = view_pose_world * O;
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
			//test_viewspace
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr test_viewspace(new pcl::PointCloud<pcl::PointXYZRGB>);
			test_viewspace->is_dense = false;
			test_viewspace->points.resize(now_view_space->views.size());
			auto pt = test_viewspace->points.begin();
			for (int i = 0; i < now_view_space->views.size(); i++, pt++) {
				(*pt).x = now_view_space->views[i].init_pos(0);
				(*pt).y = now_view_space->views[i].init_pos(1);
				(*pt).z = now_view_space->views[i].init_pos(2);
				//��һ����ʾ���е�Ϊ��ɫ
				(*pt).r = 255, (*pt).g = 255, (*pt).b = 255;
			}
			viewer->addPointCloud<pcl::PointXYZRGB>(test_viewspace, "test_viewspace");
			now_view_space->add_bbx_to_cloud(viewer);
			viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_ground_truth");
			while (!viewer->wasStopped())
			{
				viewer->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
		}

		points.clear();
		points.shrink_to_fit();
	}

	int plan() {
		switch (status)
		{
		case Over:
			break;
		case WaitData:
			if (percept->precept(now_best_view)) {
				thread next_view_space(create_view_space, &now_view_space, now_best_view, share_data, iterations);
				next_view_space.detach();
				status = WaitViewSpace;
			}
			break;
		case WaitViewSpace:
			if (share_data->now_view_space_processed) {
				thread next_views_information(create_views_information, &now_views_infromation, now_best_view, now_view_space, share_data, iterations);
				next_views_information.detach();
				status = WaitInformation;
			}
			break;
		case WaitInformation:
			if (share_data->now_views_infromation_processed) {
				if (share_data->method_of_IG == 8) { //Random
					srand(clock());
					int next_id = rand() % share_data->num_of_views; //32������
					while (now_view_space->views[next_id].vis) { //�����һ��û�з��ʹ���
						next_id = rand() % share_data->num_of_views;
					}
					now_view_space->views[next_id].vis++;
					now_view_space->views[next_id].can_move = true;
					delete now_best_view;
					now_best_view = new View(now_view_space->views[next_id]);
					cout << "choose the " << next_id << "th view." << endl;
					share_data->movement_cost += now_best_view->robot_cost;
					share_data->access_directory(share_data->save_path + "/movement");
					ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
					fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;

					//����ӵ������12/13/14/15/16��ע������в��2����Ϊһ���ǳ�ʼ�ӵ㣬����һ������ѡȡ���ӵ�
					if (iterations == 10 || iterations == 11 || iterations == 12 || iterations == 13 || iterations == 14 || iterations == (share_data->mascvp_nbv_needed_views - 2)) {
						//����һ�°�oneshot�ĳ���
						vector<int> view_set;
						for (int i = 0; i < now_view_space->views.size(); i++) {
							if (i== share_data->first_view_id) continue;
							if (now_view_space->views[i].vis) view_set.push_back(i);
						}
						View* first_view = new View(now_view_space->views[share_data->first_view_id]);
						Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data, now_view_space, first_view, view_set);
						double one_shot_distance = gloabl_path_planner->solve();
						vector<int> path_id = gloabl_path_planner->global_path;
						if (path_id.size() != iterations + 2) cout << "error: one shot path is not equal to iterations + 2" << endl;
						share_data->access_directory(share_data->save_path + "/movement_" + to_string(iterations + 2));
						double total_distance = 0;
						for (int i = 0; i < path_id.size() - 1; i++) {
							ofstream fout(share_data->save_path + "/movement_" + to_string(iterations + 2) + "/path" + to_string(i) + ".txt");
							pair<int, double> local_path = get_local_path(now_view_space->views[path_id[i]].init_pos.eval(), now_view_space->views[path_id[i+1]].init_pos.eval(), share_data->object_center_world.eval(), share_data->predicted_size * sqrt(2));
							if (local_path.first < 0) cout << "local path wrong." << endl;
							total_distance += local_path.second;
							fout << path_id[i + 1] << '\t' << local_path.second << '\t' << total_distance << endl;
						}
						// clean
						delete gloabl_path_planner;
						view_set.clear();
						view_set.shrink_to_fit();
						path_id.clear();
						path_id.shrink_to_fit();
						delete first_view;
					}
				}
				else if (share_data->method_of_IG == 6) { //NBV-NET
					share_data->access_directory(share_data->nbv_net_path + "/log");
					ifstream ftest;
					do {
						//ftest.open(share_data->nbv_net_path + "/log/ready.txt"); //���а汾
						ftest.open(share_data->nbv_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + "_ready.txt"); //���а汾
					} while (!ftest.is_open());
					ftest.close();
					ifstream fin(share_data->nbv_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + ".txt");
					int id;
					fin >> id;
					cout << "next view id is " << id << endl;
					now_view_space->views[id].vis++;
					delete now_best_view;
					now_best_view = new View(now_view_space->views[id]);
					//�˶����ۣ��ӵ�id����ǰ���ۣ��������
					share_data->movement_cost += now_best_view->robot_cost;
					share_data->access_directory(share_data->save_path + "/movement");
					ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
					fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					//���±�־�ļ�
					this_thread::sleep_for(chrono::seconds(1));
					//int removed = remove((share_data->nbv_net_path + "/log/ready.txt").c_str()); //���а汾
					int removed = remove((share_data->nbv_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + "_ready.txt").c_str()); //���а汾
					if (removed != 0) cout << "cannot remove ready.txt." << endl;
				}
				else if (share_data->method_of_IG == 9) { //PCNBV
					share_data->access_directory(share_data->pcnbv_path + "/log");
					ifstream ftest;
					do {
						//ftest.open(share_data->pcnbv_path + "/log/ready.txt"); //���а汾
						ftest.open(share_data->pcnbv_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + "_ready.txt"); //���а汾
					} while (!ftest.is_open());
					ftest.close();
					ifstream fin(share_data->pcnbv_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + ".txt");
					int id;
					fin >> id;
					cout << "next view id is " << id << endl;
					now_view_space->views[id].vis++;
					delete now_best_view;
					now_best_view = new View(now_view_space->views[id]);
					//�˶����ۣ��ӵ�id����ǰ���ۣ��������
					share_data->movement_cost += now_best_view->robot_cost;
					share_data->access_directory(share_data->save_path + "/movement");
					ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
					fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					//���±�־�ļ�
					this_thread::sleep_for(chrono::seconds(1));
					//int removed = remove((share_data->pcnbv_path + "/log/ready.txt").c_str()); //���а汾
					int removed = remove((share_data->pcnbv_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + "_ready.txt").c_str()); //���а汾
					if (removed != 0) cout << "cannot remove ready.txt." << endl;
				}
				else if (share_data->method_of_IG == 7) { //(MA-)SCVP
					if (iterations == 0 + share_data->num_of_nbvs_combined) {
						//�ӵ㸲�ǽ��
						vector<int> view_set_label;
						if (share_data->use_history_model_for_covering) {
							//��ȡNRCIP���
							share_data->access_directory(share_data->nricp_path + "/log");
							ifstream ftest;
							do {
								ftest.open(share_data->nricp_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_ready.txt"); //���а汾
							} while (!ftest.is_open());
							ftest.close();
							//��ȡ�ο�����
							pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nricp(new pcl::PointCloud<pcl::PointXYZ>);
							if (pcl::io::loadPCDFile<pcl::PointXYZ>(share_data->nricp_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_nricp.pcd", *cloud_nricp) == -1) {
								cout << "Error loading NRICP reference cloud." << endl;
							}
							cout << "NRICP reference cloud has " << cloud_nricp->points.size() << " points." << endl;
							//���Ǽ��뵱ǰ�۲������
							if (share_data->add_current_inflation) {
								//��ǰ�۲���Ƶ�kdtree
								pcl::KdTreeFLANN<pcl::PointXYZ> current_kdtree;
								pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_final(new pcl::PointCloud<pcl::PointXYZ>);
								pcl::copyPointCloud(*share_data->cloud_final, *temp_cloud_final);
								current_kdtree.setInputCloud(temp_cloud_final);
								//�ο����Ƶ�kdtree
								pcl::KdTreeFLANN<pcl::PointXYZ> nricp_kdtree;
								nricp_kdtree.setInputCloud(cloud_nricp);
								pcl::PointCloud<pcl::PointXYZ>::Ptr added_points(new pcl::PointCloud<pcl::PointXYZ>);
								for (double x = share_data->object_center_world(0) - share_data->predicted_size; x < share_data->object_center_world(0) + share_data->predicted_size; x += share_data->ground_truth_resolution * share_data->inflation_resolution_factor)
									for (double y = share_data->object_center_world(1) - share_data->predicted_size; y < share_data->object_center_world(1) + share_data->predicted_size; y += share_data->ground_truth_resolution * share_data->inflation_resolution_factor)
										for (double z = share_data->object_center_world(2) - share_data->predicted_size; z < share_data->object_center_world(2) + share_data->predicted_size; z += share_data->ground_truth_resolution * share_data->inflation_resolution_factor) {
											pcl::PointXYZ point(x, y, z);
											//�ȼ�鵱ǰ���Ƿ��뿪��ǰ�۲�����㹻��
											vector<int> pointIdxKNNSearch(1);
											vector<float> pointKNNSquaredDistance(1);
											if (current_kdtree.nearestKSearch(point, 1, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {
												if (sqrt(pointKNNSquaredDistance[0]) >= share_data->ground_truth_resolution * share_data->near_current_inflation_distance_factor) {
													continue; //�����ǰ���뿪��ǰ�۲��Դ̫Զ���򲻿���
												}
											}
											else continue; //�����ǰ���޷��ҵ���ǰ�۲�����У��򲻿���
											//���ο������Ƿ��뿪�ο������㹻Զ
											if (nricp_kdtree.nearestKSearch(point, 1, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {
												if (sqrt(pointKNNSquaredDistance[0]) <= share_data->ground_truth_resolution * share_data->far_history_inflation_distance_factor) {
													continue; //�����ǰ���뿪�ο�����̫�����򲻿���
												}
											}
											else continue; //�����ǰ���޷��ҵ��ο������У��򲻿���
											//�����ǰ�����������������ο�����
											added_points->points.push_back(point);
										}
								added_points->width = added_points->points.size();
								added_points->height = 1;
								if (added_points->points.size() > 0) {
									cout << "add " << added_points->points.size() << " points to NRICP reference cloud." << endl;
									*cloud_nricp += *added_points;
									*cloud_nricp += *temp_cloud_final; //���뵱ǰ�۲����
									//save the added points
									pcl::io::savePCDFile<pcl::PointXYZ>(share_data->save_path + "/inflation.pcd", *added_points);
									pcl::io::savePCDFile<pcl::PointXYZ>(share_data->save_path + "/current_inflation_nricp.pcd", *cloud_nricp);
								}
								else {
									cout << "no points added to NRICP reference cloud." << endl;
								}
							}
							//create a temporary share_data for rendering
							double nricp_covering_start_time = clock();
							Share_Data* share_data_temp = new Share_Data("../DefaultConfiguration.yaml", share_data->name_of_pcd);
							share_data_temp->use_saved_cloud = false; //��ʹ�ñ���ĵ���
							share_data_temp->has_table = false; //set coveringʱ����ʹ�����棬�ٶȸ���
							share_data_temp->cloud_pcd = cloud_nricp;
							share_data_temp->ground_truth_resolution = share_data->ground_truth_resolution * share_data->history_resolution_factor;
							delete share_data_temp->ground_truth_model;
							share_data_temp->ground_truth_model = new octomap::ColorOcTree(share_data_temp->ground_truth_resolution);
							//create a temporary nbv_planer
							NBV_Planner* nbv_plan_temp = new NBV_Planner(share_data_temp);
							//��ȡȫ�����ƣ���0-31˳��
							for (int i = 0; i < nbv_plan_temp->now_view_space->views.size(); i++) {
								nbv_plan_temp->percept->precept(&nbv_plan_temp->now_view_space->views[i]);
							}
							//����set covering
							set<int> sc_chosen_views;
							for (int i = 0; i < now_view_space->views.size(); i++) {
								if (now_view_space->views[i].vis)	sc_chosen_views.insert(i);
							}
							unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* all_voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
							vector<unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>*> voxels;
							for (int i = 0; i < nbv_plan_temp->now_view_space->views.size(); i++) {
								unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
								for (int j = 0; j < share_data_temp->clouds_notable[i]->points.size(); j++) {
									octomap::OcTreeKey key = share_data_temp->ground_truth_model->coordToKey(share_data_temp->clouds_notable[i]->points[j].x, share_data_temp->clouds_notable[i]->points[j].y, share_data_temp->clouds_notable[i]->points[j].z);
									if (voxel->find(key) == voxel->end()) {
										(*voxel)[key] = 1;
										//�������ظ���������Ҫ�ж��Ƿ��Ѿ�����
										if (all_voxel->find(key) != all_voxel->end()) {
											(*all_voxel)[key]++;
										}
										else {
											(*all_voxel)[key] = 1;
										}
									}
								}
								voxels.push_back(voxel);
							}
							int confidence_count = share_data->history_confidence_count;
							vector<unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>*> voxels_confident;
							for (int i = 0; i < nbv_plan_temp->now_view_space->views.size(); i++) {
								unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_confident = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
								for (auto it = voxels[i]->begin(); it != voxels[i]->end(); it++) {
									octomap::OcTreeKey key = it->first;
									if ((*all_voxel)[it->first] > confidence_count) {
										//cout << "key " << key[0] << " " << key[1] << " " << key[2] << " " << (*all_voxel)[it->first] << endl;
										(*voxel_confident)[key] = 1;
									}
								}
								voxels_confident.push_back(voxel_confident);
							}
							views_voxels_LM* SCOP_solver = new views_voxels_LM(share_data_temp, nbv_plan_temp->now_view_space, sc_chosen_views, voxels_confident);
							SCOP_solver->solve();
							view_set_label = SCOP_solver->get_view_id_set();
							//��¼ʱ��
							double nricp_covering_end_time = clock();
							double nricp_covering_time = (nricp_covering_end_time - nricp_covering_start_time) / CLOCKS_PER_SEC;
							cout << "NRICP covering time is " << nricp_covering_time << " seconds." << endl;
							ofstream fout_time(share_data->save_path + "/nricp_covering_time.txt");
							fout_time << nricp_covering_time << endl;
							fout_time.close();
							//�ͷ��ڴ�
							for (int i = 0; i < voxels.size(); i++)
								delete voxels[i];
							voxels.clear();
							voxels.shrink_to_fit();
							for (int i = 0; i < voxels_confident.size(); i++)
								delete voxels_confident[i];
							voxels_confident.clear();
							voxels_confident.shrink_to_fit();
							delete SCOP_solver;
							delete all_voxel;
							delete nbv_plan_temp;
							delete share_data_temp;
						}
						else {
							//��ȡMASCVP���
							share_data->access_directory(share_data->sc_net_path + "/log");
							ifstream ftest;
							do {
								//ftest.open(share_data->sc_net_path + "/log/ready.txt"); //���а汾
								ftest.open(share_data->sc_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_ready.txt"); //���а汾
							} while (!ftest.is_open());
							ftest.close();
							ifstream fin(share_data->sc_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + ".txt");
							int rest_view_id;
							while (fin >> rest_view_id) {
								view_set_label.push_back(rest_view_id);
							}
							//����,��ɾ���ѷ����ӵ�
							set<int> vis_view_ids;
							for (int i = 0; i < now_view_space->views.size(); i++) {
								if (now_view_space->views[i].vis)	vis_view_ids.insert(i);
							}
							for (auto it = view_set_label.begin(); it != view_set_label.end(); ) {
								if (vis_view_ids.count((*it))) {
									it = view_set_label.erase(it);
								}
								else {
									it++;
								}
							}
						}
						//����һ�������ӵ����
						ofstream fout_all_needed_views(share_data->save_path + "/all_needed_views.txt");
						fout_all_needed_views << view_set_label.size() + 1 + share_data->num_of_nbvs_combined << endl;
						cout << "All_needed_views is " << view_set_label.size() + 1 + share_data->num_of_nbvs_combined << endl;
						//���û���ӵ���Ҫ�����ֱ���˳�
						if (view_set_label.size() == 0) {
							//���±�־�ļ�
							this_thread::sleep_for(chrono::seconds(1));
							int removed = remove((share_data->sc_net_path + "/log/ready.txt").c_str());
							if (removed != 0) cout << "cannot remove ready.txt." << endl;
							//ϵͳ�˳�
							share_data->over = true;
							status = WaitMoving;
							break;
						}
						//�滮·��
						Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data, now_view_space, now_best_view, view_set_label);
						gloabl_path_planner->solve();
						//����·��
						share_data->view_label_id = gloabl_path_planner->get_path_id_set();
						delete now_best_view;
						now_best_view = new View(now_view_space->views[share_data->view_label_id[iterations - share_data->num_of_nbvs_combined]]);
						//�˶����ۣ��ӵ�id����ǰ���ۣ��������
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
						//���±�־�ļ�
						this_thread::sleep_for(chrono::seconds(1));
						if (share_data->use_history_model_for_covering) {
							int removed = remove((share_data->nricp_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_ready.txt").c_str()); //���а汾
							if (removed != 0) cout << "cannot remove ready.txt." << endl;
						}
						else {
							//int removed = remove((share_data->sc_net_path + "/log/ready.txt").c_str()); //���а汾
							int removed = remove((share_data->sc_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_ready.txt").c_str()); //���а汾
							if (removed != 0) cout << "cannot remove ready.txt." << endl;
						}
						//clean
						delete gloabl_path_planner;
						view_set_label.clear();
						view_set_label.shrink_to_fit();
					}
					else {
						if (iterations == share_data->view_label_id.size() + share_data->num_of_nbvs_combined) {
							share_data->over = true;
							status = WaitMoving;
							break;
						}
						delete now_best_view;
						now_best_view = new View(now_view_space->views[share_data->view_label_id[iterations - share_data->num_of_nbvs_combined]]);
						//�˶����ۣ��ӵ�id����ǰ���ۣ��������
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					}
				}
				else {//�����㷨
					//���ӵ�����
					sort(now_view_space->views.begin(), now_view_space->views.end(), view_utility_compare);
					/*if (share_data->sum_local_information == 0) {
						cout << "randomly choose a view" << endl;
						srand(clock());
						random_shuffle(now_view_space->views.begin(), now_view_space->views.end());
					}*/
					//informed_viewspace
					if (share_data->show) { //��ʾBBX�����λ��
						pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Iteration" + to_string(iterations)));
						viewer->setBackgroundColor(0, 0, 0);
						viewer->addCoordinateSystem(0.1);
						viewer->initCameraParameters();
						//test_viewspace
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr test_viewspace(new pcl::PointCloud<pcl::PointXYZRGB>);
						test_viewspace->is_dense = false;
						test_viewspace->points.resize(now_view_space->views.size());
						auto ptr = test_viewspace->points.begin();
						int needed = 0;
						for (int i = 0; i < now_view_space->views.size(); i++) {
							(*ptr).x = now_view_space->views[i].init_pos(0);
							(*ptr).y = now_view_space->views[i].init_pos(1);
							(*ptr).z = now_view_space->views[i].init_pos(2);
							//���ʹ��ĵ��¼Ϊ��ɫ
							if (now_view_space->views[i].vis) (*ptr).r = 0, (*ptr).g = 0, (*ptr).b = 255;
							//���������ڵ�����Ϊ��ɫ
							else if (now_view_space->views[i].in_coverage[iterations] && i < now_view_space->views.size() / 10) (*ptr).r = 255, (*ptr).g = 255, (*ptr).b = 0;
							//���������ڵ�����Ϊ��ɫ
							else if (now_view_space->views[i].in_coverage[iterations]) (*ptr).r = 255, (*ptr).g = 0, (*ptr).b = 0;
							//ǰ10%��Ȩ�صĵ�����Ϊ����ɫ
							else if (i < now_view_space->views.size() / 10) (*ptr).r = 0, (*ptr).g = 255, (*ptr).b = 255;
							//������ɫ
							else (*ptr).r = 255, (*ptr).g = 255, (*ptr).b = 255;
							ptr++;
							needed++;
						}
						test_viewspace->points.resize(needed);
						viewer->addPointCloud<pcl::PointXYZRGB>(test_viewspace, "test_viewspace");
						bool best_have = false;
						for (int i = 0; i < now_view_space->views.size(); i++) {
							if (now_view_space->views[i].vis) {
								now_view_space->views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
								Eigen::Matrix4d view_pose_world = (share_data->now_camera_pose_world * now_view_space->views[i].pose.inverse()).eval();
								Eigen::Vector4d X(0.03, 0, 0, 1);
								Eigen::Vector4d Y(0, 0.03, 0, 1);
								Eigen::Vector4d Z(0, 0, 0.03, 1);
								Eigen::Vector4d O(0, 0, 0, 1);
								X = view_pose_world * X;
								Y = view_pose_world * Y;
								Z = view_pose_world * Z;
								O = view_pose_world * O;
								viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
								viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
								viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
							}
							else if (!best_have) {
								now_view_space->views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
								Eigen::Matrix4d view_pose_world = (share_data->now_camera_pose_world * now_view_space->views[i].pose.inverse()).eval();
								Eigen::Vector4d X(0.08, 0, 0, 1);
								Eigen::Vector4d Y(0, 0.08, 0, 1);
								Eigen::Vector4d Z(0, 0, 0.08, 1);
								Eigen::Vector4d O(0, 0, 0, 1);
								X = view_pose_world * X;
								Y = view_pose_world * Y;
								Z = view_pose_world * Z;
								O = view_pose_world * O;
								viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
								viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
								viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
								best_have = true;
							}
						}
						viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_final, "cloud_now_itreation");
						while (!viewer->wasStopped())
						{
							viewer->spinOnce(100);
							boost::this_thread::sleep(boost::posix_time::microseconds(100000));
						}
					}
					double max_utility = -1;
					for (int i = 0; i < now_view_space->views.size(); i++) {
						cout << "checking view " << i << endl;
						if (now_view_space->views[i].vis) continue;
						//if (!now_view_space->views[i].can_move) continue;
						delete now_best_view;
						now_best_view = new View(now_view_space->views[i]);
						max_utility = now_best_view->final_utility;
						now_view_space->views[i].vis++;
						now_view_space->views[i].can_move = true;
						cout << "choose the " << i << "th view." << endl;
						//�˶����ۣ��ӵ�id����ǰ���ۣ��������
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
						break;
					}
					if (max_utility == -1) {
						cout << "Can't move to any viewport.Stop." << endl;
						status = Over;
						break;
					}
					cout << " next best view pos is (" << now_best_view->init_pos(0) << ", " << now_best_view->init_pos(1) << ", " << now_best_view->init_pos(2) << ")" << endl;
					cout << " next best view final_utility is " << now_best_view->final_utility << endl;
				}
				//�����˶�ģ��
				thread next_moving(move_robot, now_best_view, now_view_space, share_data);
				next_moving.detach();
				status = WaitMoving;
			}
			break;
		case WaitMoving:
			//if the method is not (combined) one-shot and random, then use f_voxel to decide whether to stop
			if(!(share_data->Combined_on == true || share_data->method_of_IG == 7 || share_data->method_of_IG == 8)){
				//compute f_voxels
				int f_voxels_num = 0;
				for (octomap::ColorOcTree::leaf_iterator it = share_data->octo_model->begin_leafs(), end = share_data->octo_model->end_leafs(); it != end; ++it) {
					double occupancy = (*it).getOccupancy();
					if (fabs(occupancy - 0.5) < 1e-3) { // unknown
						auto coordinate = it.getCoordinate();
						if (coordinate.x() >= now_view_space->object_center_world(0) - now_view_space->predicted_size && coordinate.x() <= now_view_space->object_center_world(0) + now_view_space->predicted_size
							&& coordinate.y() >= now_view_space->object_center_world(1) - now_view_space->predicted_size && coordinate.y() <= now_view_space->object_center_world(1) + now_view_space->predicted_size
							&& coordinate.z() >= now_view_space->object_center_world(2) - now_view_space->predicted_size && coordinate.z() <= now_view_space->object_center_world(2) + now_view_space->predicted_size)
						{
							// compute the frontier voxels that is unknown and has at least one free and one occupied neighbor
							int free_cnt = 0;
							int occupied_cnt = 0;
							for (int i = -1; i <= 1; i++)
								for (int j = -1; j <= 1; j++)
									for (int k = -1; k <= 1; k++)
									{
										if (i == 0 && j == 0 && k == 0) continue;
										double x = coordinate.x() + i * share_data->octomap_resolution;
										double y = coordinate.y() + j * share_data->octomap_resolution;
										double z = coordinate.z() + k * share_data->octomap_resolution;
										octomap::point3d neighbour(x, y, z);
										octomap::OcTreeKey neighbour_key;  bool neighbour_key_have = share_data->octo_model->coordToKeyChecked(neighbour, neighbour_key);
										if (neighbour_key_have) {
											octomap::ColorOcTreeNode* neighbour_voxel = share_data->octo_model->search(neighbour_key);
											if (neighbour_voxel != NULL) {
												double neighbour_occupancy = neighbour_voxel->getOccupancy();
												free_cnt += neighbour_occupancy < 0.5 ? 1 : 0;
												occupied_cnt += neighbour_occupancy > 0.5 ? 1 : 0;
											}
										}
									}
							//edge
							if (free_cnt >= 1 && occupied_cnt >= 1) {
								f_voxels_num++;
								//cout << "f voxel: " << coordinate.x() << " " << coordinate.y() << " " << coordinate.z() << endl;
							}
						}
					}
				}
				share_data->f_voxels.push_back(f_voxels_num);

				share_data->access_directory(share_data->save_path + "/f_voxels");
				ofstream fout_f_voxels_num(share_data->save_path + "/f_voxels/f_num" + to_string(iterations) + ".txt");
				fout_f_voxels_num << f_voxels_num << endl;
			    // check if the f_voxels_num is stable
				if (share_data->f_stop_iter == -1) {
					if (share_data->f_voxels.size() > 2) {
						bool f_voxels_change = false;
						//����ɨ������У���������f�仯��С����ֵ�ͽ���
						if (fabs(share_data->f_voxels[share_data->f_voxels.size() - 1] - share_data->f_voxels[share_data->f_voxels.size() - 2]) >= 32 * 32 * 32 * share_data->f_stop_threshold) {
							f_voxels_change = true;
						}
						if (fabs(share_data->f_voxels[share_data->f_voxels.size() - 2] - share_data->f_voxels[share_data->f_voxels.size() - 3]) >= 32 * 32 * 32 * share_data->f_stop_threshold) {
							f_voxels_change = true;
						}
						if (!f_voxels_change) {
							cout << "two f_voxels change smaller than threshold. Record." << endl;
							share_data->f_stop_iter = iterations;

							ofstream fout_f_stop_views(share_data->save_path + "/f_voxels/f_stop_views.txt");
							fout_f_stop_views << 1 << "\t" << share_data->f_stop_iter + 1 << endl; //1 means f_voxels stop
						}
					}
					if (share_data->over == true && share_data->f_stop_iter == -1) {
						cout << "Max iter reached. Record." << endl;
						share_data->f_stop_iter = iterations;

						ofstream fout_f_stop_views(share_data->save_path + "/f_voxels/f_stop_views.txt");
						fout_f_stop_views << 0 << "\t" << share_data->f_stop_iter + 1 << endl; //0 means over
					}
				}
				if (share_data->f_stop_iter_lenient == -1) {
					if (share_data->f_voxels.size() > 2) {
						bool f_voxels_change = false;
						//����ɨ������У���������f�仯��С����ֵ�ͽ���
						if (fabs(share_data->f_voxels[share_data->f_voxels.size() - 1] - share_data->f_voxels[share_data->f_voxels.size() - 2]) >= 32 * 32 * 32 * share_data->f_stop_threshold_lenient) {
							f_voxels_change = true;
						}
						if (fabs(share_data->f_voxels[share_data->f_voxels.size() - 2] - share_data->f_voxels[share_data->f_voxels.size() - 3]) >= 32 * 32 * 32 * share_data->f_stop_threshold_lenient) {
							f_voxels_change = true;
						}
						if (!f_voxels_change) {
							cout << "two f_voxels change smaller than threshold_lenient. Record." << endl;
							share_data->f_stop_iter_lenient = iterations;

							ofstream fout_f_stop_views(share_data->save_path + "/f_voxels/f_lenient_stop_views.txt");
							fout_f_stop_views << 1 << "\t" << share_data->f_stop_iter_lenient + 1 << endl; //1 means f_voxels stop
						}
					}
					if (share_data->over == true && share_data->f_stop_iter_lenient == -1) {
						cout << "Max iter reached. Record." << endl;
						share_data->f_stop_iter_lenient = iterations;

						ofstream fout_f_stop_views(share_data->save_path + "/f_voxels/f_lenient_stop_views.txt");
						fout_f_stop_views << 0 << "\t" << share_data->f_stop_iter_lenient + 1 << endl; //0 means over
					}
				}
			}
			//virtual move
			if (share_data->over) {
				cout << "Progress over.Saving octomap and cloud." << endl;
				status = Over;
				break;
			}
			if (share_data->move_on) {
				iterations++;
				share_data->iterations = iterations;
				share_data->now_view_space_processed = false;
				share_data->now_views_infromation_processed = false;
				share_data->move_on = false;
				status = WaitData;
			}
			break;
		}
		return status;
	}

	string out_status() {
		string status_string;
		switch (status)
		{
		case Over:
			status_string = "Over";
			break;
		case WaitData:
			status_string = "WaitData";
			break;
		case WaitViewSpace:
			status_string = "WaitViewSpace";
			break;
		case WaitInformation:
			status_string = "WaitInformation";
			break;
		case WaitMoving:
			status_string = "WaitMoving";
			break;
		}
		return status_string;
	}
};

void save_cloud_mid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string name, Share_Data* share_data) {
	//�����м�ĵ��Ƶ��̣߳�Ŀǰ������Ƿ񱣴����
	share_data->save_cloud_to_disk(cloud, "/clouds", name);
	cout << name << " saved" << endl;
	//��յ���
	cloud->points.clear();
	cloud->points.shrink_to_fit();
}

void create_view_space(View_Space** now_view_space, View* now_best_view, Share_Data* share_data, int iterations) {
	//����ؼ�֡���λ��
	now_best_view->get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
	share_data->now_camera_pose_world = (share_data->now_camera_pose_world * now_best_view->pose.inverse()).eval();;
	//����viewspace,�������Ҫ����������one-shot·���Ͳ�����OctoMap
	if (share_data->evaluate_one_shot == 0 && share_data->method_of_IG == 7 && iterations > 0 + share_data->num_of_nbvs_combined);
	else (*now_view_space)->update(iterations, share_data, share_data->cloud_final, share_data->clouds[iterations]);
	//�����м�������
	if(share_data->is_save)	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_mid(new pcl::PointCloud<pcl::PointXYZRGB>);
		*cloud_mid = *share_data->cloud_final;
		thread save_mid(save_cloud_mid, cloud_mid, "pointcloud" + to_string(iterations), share_data);
		save_mid.detach();
	}
	//���±�־λ
	share_data->now_view_space_processed = true;
}

void create_views_information(Views_Information** now_views_infromation, View* now_best_view, View_Space* now_view_space, Share_Data* share_data, int iterations) {
	if (share_data->method_of_IG == 8) { //Random
		;
	}
	else if (share_data->method_of_IG == 6) { //NBV-NET
		//octotree
		share_data->access_directory(share_data->nbv_net_path + "/data");
		ofstream fout(share_data->nbv_net_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) +'_'+ to_string(iterations) + ".txt");
		for (int i = 0; i < 32; i++)
			for (int j = 0; j < 32; j++)
				for (int k = 0; k < 32; k++)
				{
					double x = share_data->object_center_world(0) - share_data->predicted_size + share_data->octomap_resolution * i;
					double y = share_data->object_center_world(1) - share_data->predicted_size + share_data->octomap_resolution * j;
					double z = max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size) + share_data->octomap_resolution * k;
					auto node = share_data->octo_model->search(x, y, z);
					if (node == NULL) cout << "what?" << endl;
					fout << node->getOccupancy() << '\n';
				}
		fout.close();
	}
	else if (share_data->method_of_IG == 9) { //PCNBV
		share_data->access_directory(share_data->pcnbv_path + "/data");
		ofstream fout_pointcloud(share_data->pcnbv_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_pc" + to_string(iterations) + ".txt");
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);
		//�������������
		pcl::RandomSample<pcl::PointXYZRGB> ran;
		ran.setInputCloud(share_data->cloud_final);
		ran.setSample(1024); //�����²������Ƶĵ���
		ran.filter(*cloud_out);
		for (int i = 0; i < cloud_out->points.size(); i++){
			fout_pointcloud << cloud_out->points[i].x << ' '
				<< cloud_out->points[i].y << ' '
				<< cloud_out->points[i].z << '\n';
		}
		ofstream fout_viewstate(share_data->pcnbv_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_vs" + to_string(iterations) + ".txt");
		for (int i = 0; i < share_data->length_of_viewstate; i++) {
			if (now_view_space->views[i].vis)	fout_viewstate << 1 << '\n';
			else  fout_viewstate << 0 << '\n';
		}
		fout_pointcloud.close();
		fout_viewstate.close();
		cloud_out->points.clear();
		cloud_out->points.shrink_to_fit();
	}
	else if (share_data->method_of_IG == 7) { //SCVP��MA-SCVP��NRICP pipeline
		//���NBV�����ˣ�������ֱ�Ӹ���
		if (iterations == 0 + share_data->num_of_nbvs_combined) {
			//���ʹ����ʷģ�������NRICP
			if (share_data->use_history_model_for_covering) {
				share_data->access_directory(share_data->nricp_path + "/data");
				//ȥ��������ɫ��
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::copyPointCloud(*share_data->cloud_final, *cloud_out);
				//���浱ǰ�ںϹ۲�Ϊtgt����
				pcl::io::savePCDFile<pcl::PointXYZ>(share_data->nricp_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_tgt.pcd", *cloud_out);
				//������ʷģ�ͣ�Ĭ��Ϊǰһ���ģ��
				string history_daytime_string = share_data->name_of_pcd.substr(share_data->name_of_pcd.size() - 2, 2);
				int history_daytime = stoi(history_daytime_string);
				string history_model_name = share_data->name_of_pcd.substr(0, share_data->name_of_pcd.size() - 2) + to_string(history_daytime - 1);
				cout << "current model name is " << share_data->name_of_pcd << endl;
				cout << "history model name is " << history_model_name << endl;
				//��ȡ��ʷģ�ͣ������ݵ�ǰ��ת�����ת
				pcl::PointCloud<pcl::PointXYZ>::Ptr history_model(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::io::loadPCDFile<pcl::PointXYZ>(share_data->pcd_file_path + history_model_name + ".pcd", *history_model);
				//��ת��ʷģ�ͣ�ע���ȻGT��ת�ˣ������������ҲҪ��ת���������׵���NRICP��ֵ�����ⲻ����
				Eigen::Matrix3d rotation;
				rotation = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
					Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
					Eigen::AngleAxisd(45 * share_data->rotate_state * acos(-1.0) / 180.0, Eigen::Vector3d::UnitZ());
				Eigen::Matrix4d T_pose(Eigen::Matrix4d::Identity(4, 4));
				T_pose(0, 0) = rotation(0, 0); T_pose(0, 1) = rotation(0, 1); T_pose(0, 2) = rotation(0, 2); T_pose(0, 3) = 0;
				T_pose(1, 0) = rotation(1, 0); T_pose(1, 1) = rotation(1, 1); T_pose(1, 2) = rotation(1, 2); T_pose(1, 3) = 0;
				T_pose(2, 0) = rotation(2, 0); T_pose(2, 1) = rotation(2, 1); T_pose(2, 2) = rotation(2, 2); T_pose(2, 3) = 0;
				T_pose(3, 0) = 0;			   T_pose(3, 1) = 0;			  T_pose(3, 2) = 0;			     T_pose(3, 3) = 1;
				pcl::transformPointCloud(*history_model, *history_model, T_pose);
				pcl::io::savePCDFile<pcl::PointXYZ>(share_data->nricp_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_src.pcd", *history_model);
				//дconfig�ļ�
				ofstream fout_config(share_data->nricp_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_config.yaml");
				fout_config << "gpu_mode: True\n"
					<< "deformation_model: ED\n"
					<< "iters: 300\n"
					<< "lr: 0.001\n"
					<< "node_coverage: " << share_data->ground_truth_resolution * share_data->nricp_resolution_factor << "\n" //����Graph�ڵ�����ش�С
					<< "w_arap: 1\n"
					<< "w_ldmk: 0\n" //Ĭ�ϲ�ʹ��landmark�����������ǳ���Ҫ
					<< "w_chamfer: 0.1\n"
					<< "w_lap: 0.01\n"
					<< "w_silh: 0\n"
					<< "w_depth: 0\n"
					<< "src_pcd: \"" << "./data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_src.pcd\"\n"
					<< "tgt_pcd: \"" << "./data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_tgt.pcd\"\n"
					<< "use_fpfh: False\n"
					<< "voxel_size: " << share_data->ground_truth_resolution * share_data->nricp_resolution_factor << "\n" //����FPFH�����ش�С
					<< "visualize_fpfh: False\n"
					<< "visualize_nricp: False\n"
					<< "save_nricp_result: True\n"
					<< "nricp_result_path: \"" << "./log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_nricp.pcd\"\n"
					<< "cost_time_path: \"" << "./log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_nricp_time.txt\"\n";
				fout_config.close();
			}
			//�������SCVP/MA-SCVP����Ԥ��
			else { 
				//octotree
				share_data->access_directory(share_data->sc_net_path + "/data");
				//ofstream fout(share_data->sc_net_path + "/data/" + share_data->name_of_pcd + ".txt");
				ofstream fout(share_data->sc_net_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_voxel.txt");
				for (int i = 0; i < 32; i++)
					for (int j = 0; j < 32; j++)
						for (int k = 0; k < 32; k++)
						{
							double x = share_data->object_center_world(0) - share_data->predicted_size + share_data->octomap_resolution * i;
							double y = share_data->object_center_world(1) - share_data->predicted_size + share_data->octomap_resolution * j;
							double z = max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size) + share_data->octomap_resolution * k;
							auto node = share_data->octo_model->search(x, y, z);
							if (node == NULL) cout << "what?" << endl;
							fout << node->getOccupancy() << '\n';
						}
				//view state
				ofstream fout_viewstate(share_data->sc_net_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_vs.txt");
				if (share_data->MA_SCVP_on) { //�����MA-SCVP������ӵ�����
					for (int i = 0; i < share_data->length_of_viewstate; i++) {
						if (now_view_space->views[i].vis)	fout_viewstate << 1 << '\n';
						else  fout_viewstate << 0 << '\n';
					}
				}
				fout.close();
				fout_viewstate.close();
			}
		}
	}
	else { //��������
		int num_of_cover = 1;
		int num_of_voxel = 0;
		//����views_informaiton
		if (iterations == 0) (*now_views_infromation) = new Views_Information(share_data, now_view_space->voxel_information, now_view_space, iterations);
		else (*now_views_infromation)->update(share_data, now_view_space, iterations);
		if (share_data->method_of_IG == GMC) {
			//����GMC����ȡȫ���Ż�����
			views_voxels_GMC* max_cover_solver = new views_voxels_GMC(share_data->num_of_max_flow_node, now_view_space, *now_views_infromation, now_view_space->voxel_information, share_data);
			max_cover_solver->solve();
			vector<pair<int, int>> coverage_view_id_voxelnum_set = max_cover_solver->get_view_id_voxelnum_set();
			num_of_cover = coverage_view_id_voxelnum_set.size();
			for (int i = 0; i < now_view_space->views.size(); i++)
				now_view_space->views[i].in_cover = 0;
			for (int i = 0; i < coverage_view_id_voxelnum_set.size(); i++) {
				now_view_space->views[coverage_view_id_voxelnum_set[i].first].in_cover = coverage_view_id_voxelnum_set[i].second;
				num_of_voxel += coverage_view_id_voxelnum_set[i].second;
			}
			delete max_cover_solver;
			coverage_view_id_voxelnum_set.clear();
			coverage_view_id_voxelnum_set.shrink_to_fit();
			//��֤��ĸ��Ϊ0����ʵ������
			num_of_voxel = max(num_of_voxel, 1);
		}
		else if (share_data->method_of_IG == MCMF) {
			//��������������ȡȫ���Ż�����
			views_voxels_MF* set_cover_solver = new views_voxels_MF(share_data->num_of_max_flow_node, now_view_space, *now_views_infromation, now_view_space->voxel_information, share_data);
			set_cover_solver->solve();
			vector<int> coverage_view_id_set = set_cover_solver->get_view_id_set();
			for (int i = 0; i < coverage_view_id_set.size(); i++)
				now_view_space->views[coverage_view_id_set[i]].in_coverage[iterations] = 1;
			delete set_cover_solver;
			coverage_view_id_set.clear();
			coverage_view_id_set.shrink_to_fit();
		}
		//�ۺϼ���ֲ�̰����ȫ���Ż��������ӵ���Ϣ��
		share_data->sum_local_information = 0;
		share_data->sum_global_information = 0;
		share_data->sum_robot_cost = 0;
		for (int i = 0; i < now_view_space->views.size(); i++) {
			share_data->sum_local_information += now_view_space->views[i].information_gain;
			share_data->sum_global_information += now_view_space->views[i].get_global_information();
			share_data->sum_robot_cost += now_view_space->views[i].robot_cost;
		}
		//��֤��ĸ��Ϊ0����ʵ������
		if (share_data->sum_local_information == 0) share_data->sum_local_information = 1.0;
		if (share_data->sum_global_information == 0) share_data->sum_global_information = 1.0;
		for (int i = 0; i < now_view_space->views.size(); i++) {
			if (share_data->move_cost_on == false) {
				if (share_data->method_of_IG == MCMF) now_view_space->views[i].final_utility = (1 - share_data->cost_weight) * now_view_space->views[i].information_gain / share_data->sum_local_information + share_data->cost_weight * now_view_space->views[i].get_global_information() / share_data->sum_global_information;
				else if (share_data->method_of_IG == Kr) now_view_space->views[i].final_utility = now_view_space->views[i].information_gain / now_view_space->views[i].voxel_num;
				else if (share_data->method_of_IG == GMC) now_view_space->views[i].final_utility = (1 - share_data->cost_weight) * now_view_space->views[i].information_gain / share_data->sum_local_information + share_data->cost_weight * now_view_space->views[i].in_cover / num_of_voxel;
				else now_view_space->views[i].final_utility = now_view_space->views[i].information_gain;
			}
			else {
				if (share_data->method_of_IG == MCMF) now_view_space->views[i].final_utility = (1 - share_data->move_weight)* ((1 - share_data->cost_weight) * now_view_space->views[i].information_gain / share_data->sum_local_information + share_data->cost_weight * now_view_space->views[i].get_global_information() / share_data->sum_global_information) + share_data->move_weight * (share_data->robot_cost_negtive == true ? -1 : 1) * now_view_space->views[i].robot_cost / share_data->sum_robot_cost;
				else if (share_data->method_of_IG == Kr) now_view_space->views[i].final_utility = (1 - share_data->move_weight) * now_view_space->views[i].information_gain / now_view_space->views[i].voxel_num + share_data->move_weight * (share_data->robot_cost_negtive == true ? -1 : 1) * now_view_space->views[i].robot_cost / share_data->sum_robot_cost;
				else if (share_data->method_of_IG == GMC) now_view_space->views[i].final_utility = (1 - share_data->move_weight) * (1 - share_data->cost_weight) * now_view_space->views[i].information_gain / share_data->sum_local_information + share_data->cost_weight * now_view_space->views[i].in_cover / num_of_voxel + share_data->move_weight * (share_data->robot_cost_negtive == true ? -1 : 1) * now_view_space->views[i].robot_cost / share_data->sum_robot_cost;
				else now_view_space->views[i].final_utility = (1 - share_data->move_weight) * now_view_space->views[i].information_gain + share_data->move_weight * (share_data->robot_cost_negtive == true ? -1 : 1) * now_view_space->views[i].robot_cost / share_data->sum_robot_cost;
			}
		}
	}
	//���±�־λ
	share_data->now_views_infromation_processed = true;
}

void move_robot(View* now_best_view, View_Space* now_view_space, Share_Data* share_data) {
	if (share_data->iterations + 1 == share_data->num_of_nbvs_combined) { //Combined+MASCVP�л�
		share_data->method_of_IG = SCVP;
		sort(now_view_space->views.begin(), now_view_space->views.end(), view_id_compare);
	}
	if (share_data->num_of_max_iteration > 0 && share_data->iterations + 1 >= share_data->num_of_max_iteration) share_data->over = true;
	if (!share_data->move_wait) share_data->move_on = true;
}

void show_cloud(pcl::visualization::PCLVisualizer::Ptr viewer) {
	//pcl��ʾ����
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

//main.cpp
atomic<bool> stop = false;		//���Ƴ������
Share_Data* share_data;			//����������ָ��
NBV_Planner* nbv_plan;

void get_command()
{	//�ӿ���̨��ȡָ���ַ���
	string cmd;
	while (!stop && !share_data->over)
	{
		cout << "Input command 1.stop 2.over 3.next_itreation :" << endl;
		cin >> cmd;
		if (cmd == "1") stop = true;
		else if (cmd == "2") share_data->over = true;
		else if (cmd == "3") share_data->move_on = true;
		else cout << "Wrong command.Retry :" << endl;
	}
	cout << "get_command over." << endl;
}

void get_run()
{
	//NBV�滮�ڳ�ʼ��
	nbv_plan = new NBV_Planner(share_data);
	//����ѭ��
	string status="";
	//ʵʱ��ȡ��滮
	while (!stop && nbv_plan->plan()) {
		//���״̬�б仯�����
		if (status != nbv_plan->out_status()) {
			status = nbv_plan->out_status();
			cout << "NBV_Planner's status is " << status << endl;
		}
	}
	delete nbv_plan;
}

#define GetGTPoints 0
#define DebugOne 1
#define TestAll 2
#define POCOMesh 3
#define HandGTSize 4
#define GetGTNBV 5

int mode = GetGTPoints;

int main()
{
	//Init
	ios::sync_with_stdio(false);
	cout << "input mode:";
	cin >> mode;

	vector<int> rotate_states;
	rotate_states.push_back(0);
	rotate_states.push_back(1);
	//rotate_states.push_back(2);
	//rotate_states.push_back(3);
	//rotate_states.push_back(4);
	//rotate_states.push_back(5);
	//rotate_states.push_back(6);
	//rotate_states.push_back(7);

	vector<int> first_view_ids;
	//first_view_ids.push_back(0);
	//first_view_ids.push_back(1);
	//first_view_ids.push_back(2);
	//first_view_ids.push_back(3);
	//first_view_ids.push_back(4);
	//first_view_ids.push_back(5);
	//first_view_ids.push_back(6);
	//first_view_ids.push_back(7);
	first_view_ids.push_back(8);
	//first_view_ids.push_back(9);
	//first_view_ids.push_back(10);
	//first_view_ids.push_back(11);
	//first_view_ids.push_back(12);
	//first_view_ids.push_back(13);
	//first_view_ids.push_back(14);
	//first_view_ids.push_back(15);
	//first_view_ids.push_back(16);
	first_view_ids.push_back(17);
	//first_view_ids.push_back(18);
	//first_view_ids.push_back(19);
	//first_view_ids.push_back(20);
	first_view_ids.push_back(21);
	//first_view_ids.push_back(22);
	//first_view_ids.push_back(23);
	//first_view_ids.push_back(24);
	//first_view_ids.push_back(25);
	//first_view_ids.push_back(26);
	//first_view_ids.push_back(27);
	//first_view_ids.push_back(28);
	//first_view_ids.push_back(29);
	//first_view_ids.push_back(30);
	//first_view_ids.push_back(31);

	int combined_test_on;
	cout << "combined on:";
	cin >> combined_test_on;

	////scvp��7��Ҫ���ܣ�����������ȡ�����������ܾͰ�Ĭ��ֵ
	//vector<int> methods;
	//methods.push_back(7);
	//methods.push_back(3);
	//methods.push_back(4);
	//methods.push_back(0);
	//methods.push_back(6);

	int method_id;
	cout << "thread for method id:";
	cin >> method_id;
	int move_test_on;
	cout << "move test on :";
	cin >> move_test_on;
	//���Լ�
	vector<string> names;
	cout << "input models:" << endl;
	string name;
	while (cin >> name) {
		if (name == "-1") break;
		names.push_back(name);
	}
	//ѡȡģʽ
	if (mode == DebugOne)
	{
		//��������ʼ��
		share_data = new Share_Data("../DefaultConfiguration.yaml");
		//����̨��ȡָ���߳�
		thread cmd(get_command);
		//NBVϵͳ�����߳�
		thread runner(get_run);
		//�ȴ��߳̽���
		runner.join();
		cmd.join();
		delete share_data;
	}
	else if (mode == TestAll) {
		//�����������塢�ӵ㡢����
		for (int i = 0; i < names.size(); i++) {
			for (int j = 0; j < rotate_states.size(); j++) {
				for (int k = 0; k < first_view_ids.size(); k++) {
					//��������ʼ��
					share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], rotate_states[j], first_view_ids[k], method_id, move_test_on, combined_test_on);
					//NBVϵͳ�����߳�
					thread runner(get_run);
					//�ȴ��߳̽���
					runner.join();
					delete share_data;
				}
			}
		}
	}
	else if (mode == GetGTPoints) {
		for (int i = 0; i < names.size(); i++) {
			cout << "Get GT visible pointcloud number of model " << names[i] << endl;
			
			/*
			// ��ȡ����ߴ�
			share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], 0);
			nbv_plan = new NBV_Planner(share_data);
			ofstream write;
			write.open("D:\\����\\MA-SCVP\\size.txt", ios::app);
			write << share_data->predicted_size << endl;
			write.close();
			*/

			//for (int rotate_state = 0; rotate_state < 8; rotate_state++) {
			for (int x = 0; x < rotate_states.size(); x++) {
				int rotate_state = rotate_states[x];
				//��������ʼ��
				share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], rotate_state);
				//NBV�滮�ڳ�ʼ��
				nbv_plan = new NBV_Planner(share_data);

				/*
				// ��ȡ�ӵ�ռ��ؽ���
				ifstream read("D:\\Data\\MA-SCVP\\GT_points_num\\" + names[i] + "_r" + to_string(rotate_state) + ".txt");
				int SC16, SC32, SC64;
				read >> SC32;
				read.close();
				read.open("D:\\Data\\MA-SCVP\\GT_points_num\\\\GT_points_num_64\\" + names[i] + "_r" + to_string(rotate_state) + ".txt");
				read >> SC64;
				read.close();
				read.open("D:\\Data\\MA-SCVP\\GT_points_num\\\\GT_points_num_16\\" + names[i] + "_r" + to_string(rotate_state) + ".txt");
				read >> SC16;
				ofstream write;
				write.open("D:\\����\\MA-SCVP\\SC_viewspace.txt", ios::app);
				write << names[i] + "_r" + to_string(rotate_state) << "\t" << SC16 << "\t" << SC32 << "\t" << SC64 << "\t" << share_data->cloud_points_number << endl;
				write.close();
				*/
				
				//��ȡȫ�����ƣ���0-31˳��
				for (int i = 0; i < nbv_plan->now_view_space->views.size(); i++) {
					nbv_plan->percept->precept(&nbv_plan->now_view_space->views[i]);
				}
				//����Octomap����ͳ�ƿɼ������
				double now_time = clock();
				int num = 0;
				unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
				for (int j = 0; j < share_data->cloud_final->points.size(); j++) {
					octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(share_data->cloud_final->points[j].x, share_data->cloud_final->points[j].y, share_data->cloud_final->points[j].z);
					if (voxel->find(key) == voxel->end()) {
						(*voxel)[key] = num++;
					}
				}
				cout << "GT visible pointcloud get with executed time " << clock() - now_time << " ms." << endl;
				/*
				for (octomap::ColorOcTree::leaf_iterator it = nbv_plan->percept->ground_truth_model->begin_leafs(), end = nbv_plan->percept->ground_truth_model->end_leafs(); it != end; ++it) {
					if (voxel->find(it.getKey()) != voxel->end()) {
						nbv_plan->percept->ground_truth_model->setNodeColor(it.getKey(), 255, 0, 0);
					}
				}
				nbv_plan->percept->ground_truth_model->write("D:\\����\\MA-SCVP\\inner.ot");
				*/
				
				//����GT
				share_data->access_directory(share_data->gt_path + "/GT_points/" + names[i] + "_r" + to_string(rotate_state) + "/");

				for (int j = 0; j < nbv_plan->now_view_space->views.size(); j++) {
					//�������
					pcl::io::savePCDFile<pcl::PointXYZRGB>(share_data->gt_path + "/GT_points/" + names[i] + "_r" + to_string(rotate_state) + "/cloud_view" + to_string(j) + ".pcd", *share_data->clouds[j]);
					pcl::io::savePCDFile<pcl::PointXYZRGB>(share_data->gt_path + "/GT_points/" + names[i] + "_r" + to_string(rotate_state) + "/cloud_notable_view" + to_string(j) + ".pcd", *share_data->clouds_notable[j]);
				}
				ofstream fout_GT_points_num(share_data->gt_path + "/GT_points/" + names[i] + "_r" + to_string(rotate_state) + "/visible_num.txt");
				fout_GT_points_num << voxel->size() << endl;
				cout << "Rotate " << rotate_state << " GT_points_num is " << voxel->size() << " ,rate is " << 1.0 * voxel->size() / share_data->cloud_points_number << endl;
				delete voxel;
				delete nbv_plan;
				delete share_data;
				
			}
			
		}
	}
	else if (mode == POCOMesh) {
		for (int i = 0; i < names.size(); i++) {
			for (int j = 0; j < rotate_states.size(); j++) {
				for (int k = 0; k < first_view_ids.size(); k++) {
					share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], rotate_states[j], first_view_ids[k], method_id, move_test_on, combined_test_on);
					cout << "now save path is: " << share_data->save_path << endl;
					nbv_plan = new NBV_Planner(share_data);
					string method_str = to_string(share_data->method_of_IG) + "_" + to_string(share_data->num_of_nbvs_combined);
					string output_path = share_data->gt_path + "Finals/" + method_str;
					output_path += "/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id);
					share_data->access_directory(output_path);
					//��ѡȡ���ӵ�
					vector<int> chosen_views;
					ifstream fin_all_needed_views(share_data->save_path + "/all_needed_views.txt");
					int num_of_all_needed_views = -1;
					if (fin_all_needed_views.is_open()) {
						fin_all_needed_views >> num_of_all_needed_views;
					}
					else {
						num_of_all_needed_views = share_data->num_of_max_iteration;
					}
					chosen_views.push_back(share_data->first_view_id);
					for (int i = 0; i < num_of_all_needed_views - 1; i++) {
						ifstream fin_view_id(share_data->save_path + "/movement/path" + to_string(i) + ".txt");
						int view_id;
						fin_view_id >> view_id;
						chosen_views.push_back(view_id);
						//cout << view_id << endl;
					}
					if (chosen_views.size() != num_of_all_needed_views) cout << "Error!" << endl;

					//���һ��BBX
					share_data->access_directory(share_data->gt_path + "bbx");
					ifstream fin_bbx_2mm(share_data->gt_path + "bbx/" + share_data->name_of_pcd + ".txt");
					bool bbx_2mm_open = fin_bbx_2mm.is_open();
					ofstream fout_bbx_2mm;
					if (!bbx_2mm_open) fout_bbx_2mm.open(share_data->gt_path + "bbx/" + share_data->name_of_pcd + ".txt");
					for (double x = share_data->object_center_world(0) - share_data->predicted_size; x < share_data->object_center_world(0) + share_data->predicted_size; x += share_data->ground_truth_resolution)
						for (double y = share_data->object_center_world(1) - share_data->predicted_size; y < share_data->object_center_world(1) + share_data->predicted_size; y += share_data->ground_truth_resolution)
							for (double z = max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size); z < max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size) + 2.0 * share_data->predicted_size; z += share_data->ground_truth_resolution) {
								octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(octomap::point3d(x, y, z));
								octomap::point3d coord = share_data->ground_truth_model->keyToCoord(key);
								if (!bbx_2mm_open) fout_bbx_2mm << coord.x() << ' ' << coord.y() << ' ' << coord.z() << '\n';
							}

					//����ֱ��ʸߵĳ���
					octomap::ColorOcTree* octo_model_whole_2mm = new octomap::ColorOcTree(share_data->ground_truth_resolution);
					for (double x = share_data->object_center_world(0) - share_data->predicted_size; x < share_data->object_center_world(0) + share_data->predicted_size; x += share_data->ground_truth_resolution)
						for (double y = share_data->object_center_world(1) - share_data->predicted_size; y < share_data->object_center_world(1) + share_data->predicted_size; y += share_data->ground_truth_resolution)
							for (double z = max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size); z < max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size) + 2.0 * share_data->predicted_size; z += share_data->ground_truth_resolution) {
								octo_model_whole_2mm->setNodeValue(x, y, z, (float)0, true); //��ʼ������0.5����logoddsΪ0
							}
					octo_model_whole_2mm->updateInnerOccupancy();

					//����ÿ�ε����ĵ�������ѵ��
					for (int iteration = 0; iteration < chosen_views.size(); iteration++) {
						//��ȡѡȡ�ӵ�ĵ���
						nbv_plan->percept->precept(&nbv_plan->now_view_space->views[chosen_views[iteration]]);
						//update OctoMap ������
						octomap::Pointcloud cloud_octo;
						for (auto& p : share_data->clouds[iteration]->points) {
							cloud_octo.push_back(p.x, p.y, p.z);
						}
						//ע��˴�cloud�ĵ�i����Ӧ���ӵ���chosen_views[i]
						octo_model_whole_2mm->insertPointCloud(cloud_octo, octomap::point3d(nbv_plan->now_view_space->views[chosen_views[iteration]].init_pos(0), nbv_plan->now_view_space->views[chosen_views[iteration]].init_pos(1), nbv_plan->now_view_space->views[chosen_views[iteration]].init_pos(2)), -1, true, false);
						for (auto& p : share_data->clouds[iteration]->points) {
							octo_model_whole_2mm->integrateNodeColor(p.x, p.y, p.z, 255, 0, 0);
							if (p.z < share_data->min_z_table + share_data->octomap_resolution) octo_model_whole_2mm->setNodeColor(p.x, p.y, p.z, 0, 0, 255);
						}
						octo_model_whole_2mm->updateInnerOccupancy();
						//octo_model_whole_2mm->write(output_path + "/" + to_string(iteration) + "/scene.ot");
						//cout << "octo_model getted." << endl;
					}

					//get voxel map ��������
					int full_num = 0;
					unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* all_voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
					for (int j = 0; j < share_data->cloud_final->points.size(); j++) {
						octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(share_data->cloud_final->points[j].x, share_data->cloud_final->points[j].y, share_data->cloud_final->points[j].z);
						if (all_voxel->find(key) == all_voxel->end()) {
							(*all_voxel)[key] = full_num++;
						}
					}
					//cout << "full_num is " << full_num << endl;

					share_data->access_directory(output_path + "/");
					ofstream fout_full_surface(output_path + "/surface.txt");
					ofstream fout_full_scene(output_path + "/scene.txt");
					//д��PCL����
					double min_z = 10.0;
					for (auto it = all_voxel->begin(); it != all_voxel->end(); it++) {
						octomap::OcTreeKey key = it->first;
						octomap::point3d coord = share_data->ground_truth_model->keyToCoord(key);
						if (coord.z() < min_z) min_z = coord.z();
					}
					//cout << "share_data->min_z_table: " << share_data->min_z_table << endl;
					//cout << "min_z: " << min_z << endl;
					double min_x_table = 10.0, max_x_table = -10.0, min_y_table = 10.0, max_y_table = -10.0;
					pcl::PointCloud<pcl::PointXYZ>::Ptr all_cloud(new pcl::PointCloud<pcl::PointXYZ>);
					for (auto it = all_voxel->begin(); it != all_voxel->end(); it++) {
						octomap::OcTreeKey key = it->first;
						octomap::point3d coord = share_data->ground_truth_model->keyToCoord(key);
						fout_full_surface << coord.x() << ' ' << coord.y() << ' ' << coord.z() << "\n";
						fout_full_scene << coord.x() << ' ' << coord.y() << ' ' << coord.z() << ' ' << 1 << "\n";
						pcl::PointXYZ p;
						p.x = coord.x();
						p.y = coord.y();
						p.z = coord.z();
						all_cloud->push_back(p);
						if (coord.z() <= min_z + 1e-6) {
							min_x_table = min(min_x_table, (double)coord.x());
							max_x_table = max(max_x_table, (double)coord.x());
							min_y_table = min(min_y_table, (double)coord.y());
							max_y_table = max(max_y_table, (double)coord.y());
						}
					}
					//cout << "min_x_table: " << min_x_table << ",max_x_table: " << max_x_table << endl;
					//cout << "min_y_table: " << min_y_table << ",max_y_table: " << max_y_table << endl;
					vector<int> pointIdxKNNSearch(1);
					vector<float> pointKNNSquaredDistance(1);
					pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
					kdtree.setInputCloud(all_cloud);

					for (double x = share_data->object_center_world(0) - share_data->predicted_size; x < share_data->object_center_world(0) + share_data->predicted_size; x += share_data->ground_truth_resolution)
						for (double y = share_data->object_center_world(1) - share_data->predicted_size; y < share_data->object_center_world(1) + share_data->predicted_size; y += share_data->ground_truth_resolution)
							for (double z = max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size); z < max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size) + 2.0 * share_data->predicted_size; z += share_data->ground_truth_resolution) {
								octomap::OcTreeKey key = octo_model_whole_2mm->coordToKey(octomap::point3d(x, y, z));
								octomap::point3d coord = octo_model_whole_2mm->keyToCoord(key);
								//��������������
								if (coord.z() < min_z - share_data->ground_truth_resolution - 1e-6) {
									continue;
								}
								else if (coord.z() < min_z - 1e-6) {
									if (coord.x() > max_x_table + 1e-6 || coord.x() < min_x_table - 1e-6) {
										continue;
									}
									if (coord.y() > max_y_table + 1e-6 || coord.y() < min_y_table - 1e-6) {
										continue;
									}
									fout_full_scene << coord.x() << ' ' << coord.y() << ' ' << coord.z() << ' ' << 0 << '\n';
								}
								else { //����������棬���������С��һ����ֵ�ĵ�
									//��all_cloud����������Ϊxxx�ĵ�
									pcl::PointXYZ search_point;
									search_point.x = coord.x();
									search_point.y = coord.y();
									search_point.z = coord.z();
									if (kdtree.nearestKSearch(search_point, 1, pointIdxKNNSearch, pointKNNSquaredDistance) <= 0) {
										continue;
									}
									//cout<< coord.x() << ' ' << coord.y() << ' ' << coord.z() << ' ' << sqrt(pointKNNSquaredDistance[0]) <<endl;
									//�������˾����㹻��
									if (sqrt(pointKNNSquaredDistance[0]) > 2.0 * share_data->ground_truth_resolution) {
										continue;
									}
									auto node = octo_model_whole_2mm->search(x, y, z);
									int is_surface = -1;
									if (node->getOccupancy() > 0.5) is_surface = 1;
									else if (node->getOccupancy() < 0.5) is_surface = 0;
									//����surface
									if (is_surface == 0) { //�۲�Ŀյ�ȫ0
										fout_full_scene << coord.x() << ' ' << coord.y() << ' ' << coord.z() << ' ' << 0 << '\n';
									}
								}
							}
					all_cloud->points.clear();
					all_cloud->points.shrink_to_fit();
					fout_full_surface.close();
					fout_full_scene.close();
					delete all_voxel;

					chosen_views.clear();
					chosen_views.shrink_to_fit();
					delete octo_model_whole_2mm;
					delete nbv_plan;
					delete share_data;
					//system("pause");
				}
			}
		}
	}
	else if (mode == HandGTSize) {
		//Shrinking the point cloud of an object makes the surface watertight
		for (int i = 0; i < names.size(); i++) {
			cout << "Get GT size of pointcloud model " << names[i] << endl;
			share_data = new Share_Data("../DefaultConfiguration.yaml", names[i]);
			nbv_plan = new NBV_Planner(share_data);
			double original_size = nbv_plan->now_view_space->predicted_size;
			share_data->access_directory(share_data->pre_path + "/Ground_size/");
			for (auto it = share_data->ground_truth_model->begin_leafs(); it != share_data->ground_truth_model->end_leafs(); it++) {
				share_data->ground_truth_model->setNodeColor(it.getKey(), 255, 0, 0);
			}
			share_data->ground_truth_model->write(share_data->pre_path + "/Ground_size/" + names[i] + ".ot");
			double minus_size = 0.0;
			bool ok;
			cout << "Minus 0.01m (1/0):";
			cin >> ok;
			while (ok) {
				minus_size += 0.01;
				delete nbv_plan;
				delete share_data;
				share_data = new Share_Data("../DefaultConfiguration.yaml", names[i]);
				share_data->mp_scale[share_data->name_of_pcd] = minus_size;
				nbv_plan = new NBV_Planner(share_data);
				share_data->access_directory(share_data->pre_path + "/Ground_size/");
				for (auto it = share_data->ground_truth_model->begin_leafs(); it != share_data->ground_truth_model->end_leafs(); it++) {
					share_data->ground_truth_model->setNodeColor(it.getKey(), 255, 0, 0);
				}
				share_data->ground_truth_model->write(share_data->pre_path + "/Ground_size/" + names[i] + ".ot");
				cout << "Minus 0.01m (1/0):";
				cin >> ok;
			};
			if (minus_size != 0) {
				share_data->access_directory(share_data->pre_path + "/Ground_size/");
				ofstream fout_GT_size(share_data->pre_path + "/Ground_size/" + names[i] + ".txt");
				fout_GT_size << minus_size << '\n';
			}
		}
	}
	else if (mode == GetGTNBV) {
		for (int name_id = 0; name_id < names.size(); name_id++) {
			for (int rotate_id = 0; rotate_id < rotate_states.size(); rotate_id++) {
				for (int first_view_idx = 0; first_view_idx < first_view_ids.size(); first_view_idx++) {
					share_data = new Share_Data("../DefaultConfiguration.yaml", names[name_id], rotate_states[rotate_id], first_view_ids[first_view_idx], method_id, move_test_on, combined_test_on);
					nbv_plan = new NBV_Planner(share_data);
					//��ȡȫ�����ƣ���0-31˳��
					for (int i = 0; i < nbv_plan->now_view_space->views.size(); i++) {
						nbv_plan->percept->precept(&nbv_plan->now_view_space->views[i]);
					}
					//����set covering
					set<int> sc_chosen_views;
					sc_chosen_views.insert(share_data->first_view_id);
					unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* all_voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
					vector<unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>*> voxels;
					for (int i = 0; i < nbv_plan->now_view_space->views.size(); i++) {
						unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
						for (int j = 0; j < share_data->clouds_notable[i]->points.size(); j++) {
							octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(share_data->clouds_notable[i]->points[j].x, share_data->clouds_notable[i]->points[j].y, share_data->clouds_notable[i]->points[j].z);
							if (voxel->find(key) == voxel->end()) {
								(*voxel)[key] = 1;
								//�������ظ���������Ҫ�ж��Ƿ��Ѿ�����
								if (all_voxel->find(key) != all_voxel->end()) {
									(*all_voxel)[key]++;
								}
								else {
									(*all_voxel)[key] = 1;
								}
							}
						}
						voxels.push_back(voxel);
					}
					int confidence_count = 1;
					vector<int> num_hits;
					num_hits.resize(nbv_plan->now_view_space->views.size() + 1);
					for (int i = 1; i <= nbv_plan->now_view_space->views.size(); i++) {
						num_hits[i] = 0;
					}
					for (auto it = all_voxel->begin(); it != all_voxel->end(); it++) {
						num_hits[(*all_voxel)[it->first]]++;
					}
					//for (int i = 1; i <= nbv_plan->now_view_space->views.size(); i++) {
					//	cout<< "hits " << i << " have " << num_hits[i] << endl;
					//}
					vector<unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>*> voxels_confident;
					for (int i = 0; i < nbv_plan->now_view_space->views.size(); i++) {
						unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_confident = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
						for (auto it = voxels[i]->begin(); it != voxels[i]->end();it++) {
							octomap::OcTreeKey key = it->first;
							if ((*all_voxel)[it->first] > confidence_count) {
								//cout << "key " << key[0] << " " << key[1] << " " << key[2] << " " << (*all_voxel)[it->first] << endl;
								(*voxel_confident)[key] = 1;
							}
						}
						voxels_confident.push_back(voxel_confident);
					}
					//cout << "All voxels num is " << all_voxel->size() << endl;
					//for (auto it = all_voxel->begin(); it != all_voxel->end();) {
					//	octomap::OcTreeKey key = it->first;
					//	if ((*all_voxel)[it->first] <= confidence_count) {
					//		it = all_voxel->erase(it);
					//	}
					//	else {
					//		it++;
					//	}
					//}
					//cout << "confidence_all_voxel size " << all_voxel->size() << endl;
					views_voxels_LM* SCOP_solver = new views_voxels_LM(share_data, nbv_plan->now_view_space, sc_chosen_views, voxels_confident);
					SCOP_solver->solve();
					vector<int> need_views = SCOP_solver->get_view_id_set();
					//������
					string out_path = share_data->pre_path + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id);
					share_data->access_directory(out_path);
					ofstream out_sc(out_path + "/set_covering.txt");
					ofstream out_num_sc(out_path + "/all_needed_covering.txt");
					out_num_sc << need_views.size() << '\n';
					for (int i = 0; i < need_views.size(); i++) {
						out_sc << need_views[i] << '\n';
					}
					out_sc.close();
					out_num_sc.close();
					//�ͷ��ڴ�
					for (int i = 0; i < voxels.size(); i++)
						delete voxels[i];
					voxels.clear();
					voxels.shrink_to_fit();
					for (int i = 0; i < voxels_confident.size(); i++)
						delete voxels_confident[i];
					voxels_confident.clear();
					voxels_confident.shrink_to_fit();
					delete SCOP_solver;
					delete all_voxel;
					delete nbv_plan;
					delete share_data;
				}
			}
		}
	}
	cout << "System over." << endl;
	return 0;
}

/*
maize_plant1_03-21
maize_plant2_03-21
maize_plant3_03-21
tomato_plant1_03-13
tomato_plant2_03-13
tomato_plant3_03-13
*/
