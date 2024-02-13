#pragma once

#include "utility.h"

static struct SkeletConnParams {
  int norm_len = 5;
  int step = 10;
  // int st_num = 651;
  int st_num = 10;
  int fin_num = 2676;
  // int dif = 0;
  int long_path = 15;
  int long_dist = 100;
  int long_path_cont = 100;
  int near_to_fin = 10;
  int big_cycle_begin = 150;
  int folded001_st_seg_len = 60;
  int long_path_st_slice = 60;
  int slice_count = 2687;
  int big_cycle_middle = 100;
  int curvature_dif_len = 10;
  int len_est_long_path = 100;
  int len_est_beg_mul = 5;
  int len_est_mid_mul = 3;
  int twisting_dist = 15;
  int twisting_dif_len = 100;

} scroll_const;

void delete_skelet_borders(cv::Mat &skelet);

void delete_big_cycles(cv::Mat &skelet, FindCompResult &comp_res, bool begin);

void delete_branch_points(FindCompResult &comp_res, cv::Mat &skelet,
                          bool st_slice);

std::vector<std::vector<cv::Point>> collect_paths(FindCompResult &comp_res,
                                                  bool st_slice, bool debug);

PathPos find_st_path_st_slice(std::vector<std::vector<cv::Point>> &paths,
                              ScrollType tp);

PathPos find_st_path_mid_slice(std::vector<std::vector<cv::Point>> &paths,
                               std::vector<std::pair<double, int>> &dst_idx,
                               std::vector<cv::Point> &big_path);

std::pair<bool, std::vector<PathPos>>
collect_nearest_paths_st_slice(std::vector<cv::Point> &big_path,
                               std::vector<std::vector<cv::Point>> &paths,
                               std::vector<bool> &us_pth);

std::pair<bool, std::vector<PathPos>> collect_nearest_paths_mid_slice(
    std::vector<cv::Point> &big_path,
    std::vector<std::vector<cv::Point>> &paths, std::vector<bool> &us_pth,
    std::vector<cv::Point> &pred_pts_raw, std::vector<cv::Point2d> &pred_pts,
    bool debug);

void st_slice_choose_best_cand(std::vector<PathPos> &idx_pos,
                               std::vector<std::vector<cv::Point>> &paths,
                               std::vector<cv::Point> &big_path, cv::Mat &raw,
                               std::vector<bool> &us_pth);

class CompSkeletsBackups {

public:
  bool operator()(PathPos p1, PathPos p2);

public:
  std::vector<std::vector<cv::Point>> paths;
  std::vector<std::vector<double>> paths_dist;
  std::vector<cv::Point2d> pred_pts;
  std::vector<double> pref_dst;
  int cur_min;
  std::vector<std::pair<double, int>> dst_idx;
};

void draw_result_st(cv::Mat &img, std::vector<cv::Point> &big_path,
                    std::string out_path, int st_num, std::string scroll_id);

void texturing_operation(cv::Mat &fst_img, cv::Mat &sec_img,
                         int slice_num, std::vector<cv::Point2d> &fst_path,
                         bool debug, std::string raw_path,
                         std::string out_path, const std::set<int>& nums, std::string scroll_id);
/*
void load_from_checkpoint(int &i, int &checkpoint_num, std::string nm,
                          std::vector<cv::Point2d> &pred_pts,
                          std::vector<cv::Point> &pred_pts_raw,
                          double &tot_dist, std::vector<cv::Point> &big_path,
                          std::string out_path, cv::Mat &fst_img,
                          cv::Mat &sec_img);
*/
std::vector<std::vector<double>>
calc_pref_dist(std::vector<std::vector<cv::Point>> &paths);

std::vector<std::pair<double, int>>
calc_min_dist(std::vector<std::vector<cv::Point>> &paths,
              std::vector<cv::Point2d> &pred_pts);

int calc_twisting(std::vector<cv::Point2d> &pred_pts);

bool case_mid_nearest(std::vector<std::vector<cv::Point>> &paths,
                      std::map<cv::Point, int, CompClass> &map_pts,
                      double &tot_dist, int &cur_min,
                      std::vector<cv::Point> &big_path, PathOrder pos, int idx,
                      std::vector<std::vector<double>> paths_dist);

void skelet_conn_new(std::vector<std::string> raw_paths, bool debug, bool swap_order,
                    std::string out_path_details, std::vector<std::string> mask_paths, std::vector<int> nums, int& st_idx, int fin_idx,
                    int slice_count, std::string scroll_id);

void write_pred_pts(std::vector<cv::Point2d> &pred_pts,
                    std::vector<cv::Point> &pred_pts_raw, int st_num,
                    std::string out_path, std::string scroll_id);
