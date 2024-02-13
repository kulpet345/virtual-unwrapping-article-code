#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <nlohmann/json.hpp>
#include <filesystem>


enum PathOrder { direct, reverse };

struct PathPos {
  int idx;
  PathOrder order;
};

struct CompClass {
  bool operator()(const cv::Point &pt1, const cv::Point &pt2) const {
    return (pt1.x < pt2.x) || (pt1.x == pt2.x && pt1.y < pt2.y);
  }
};

struct CompClass3 {
  bool operator()(const cv::Point3i &pt1, const cv::Point3i &pt2) const {
    return (pt1.x < pt2.x) || (pt1.x == pt2.x && pt1.y < pt2.y) ||
           (pt1.x == pt2.x && pt1.y == pt2.y && pt1.z < pt2.z);
  }
};

struct FindCompResult {
  int cnt;
  std::vector<std::vector<int>> used;
  std::vector<std::vector<cv::Point>> comps;
};

enum ScrollType {
  scroll001,
  scroll002,
  scroll003,
  scroll004,
  folded001,
  folded002,
  notes
};


struct ConfigParams {
  std::string scroll_id;
  std::string folder_path_slices;
  double scale_coef;
  bool flip_vert_unfolding_bool;
  bool flip_hori_unfolding_bool;
  bool rotate_90_unfolding_bool;
  std::string file_path_num_path;
  std::string file_path_scan_3d;
  std::string folder_path_mask;
  std::string folder_path_final;
  bool write_unnecessary_details;
  std::string folder_path_details;
};

std::string get_num(int n, int k = 4);

std::vector<cv::Point2d> gaus_smooth(std::vector<cv::Point2d> pts);

void dfs_comp(int x, int y, const cv::Mat &img,
              std::vector<std::vector<int>> &out_comp, int cnt);

template <typename T>
std::vector<cv::Point2d> uniform_sampling(std::vector<T> pts, int cnt_fin) {
  std::vector<cv::Point2d> int_pts;
  double dist = 0;
  for (int i = 1; i < pts.size(); ++i) {
    dist += sqrt((pts[i].x - pts[i - 1].x) * (pts[i].x - pts[i - 1].x) +
                 (pts[i].y - pts[i - 1].y) * (pts[i].y - pts[i - 1].y));
  }
  int curv = 0;
  double curdl = 0;
  double curdr = sqrt((pts[1].x - pts[0].x) * (pts[1].x - pts[0].x) +
                      (pts[1].y - pts[0].y) * (pts[1].y - pts[0].y));
  int idx = 1;
  for (int i = 0; i < cnt_fin; ++i) {
    double sd = dist / (cnt_fin - 1) * i;
    while (curdr < sd && idx + 1 < pts.size()) {
      curdl = curdr;
      curdr +=
          sqrt((pts[idx + 1].x - pts[idx].x) * (pts[idx + 1].x - pts[idx].x) +
               (pts[idx + 1].y - pts[idx].y) * (pts[idx + 1].y - pts[idx].y));
      ++idx;
    }
    double dif = sd - curdl;
    cv::Point2d dbs;
    dbs.x =
        pts[idx - 1].x + dif / (curdr - curdl) * (pts[idx].x - pts[idx - 1].x);
    dbs.y =
        pts[idx - 1].y + dif / (curdr - curdl) * (pts[idx].y - pts[idx - 1].y);
    int_pts.push_back(dbs);
  }
  return int_pts;
}

std::vector<cv::Point2d> resample_fixed(std::vector<cv::Point2d> pts,
                                        double step);

double find_median(cv::Mat input);

double find_quantile(cv::Mat input, double quantile);

cv::Mat image_contrast(cv::Mat img);

cv::Mat gaus_filter(cv::Mat img, int kernel_size);

cv::Mat image_to_int(cv::Mat img);

cv::Mat read_tif(std::string path, int fl = 32);

void dfs_comp1(const cv::Mat &img, std::vector<std::vector<int>> &used,
               cv::Point pt, int cur, std::vector<cv::Point> &all_pt);

std::vector<cv::Point>
get_neighbours(cv::Point pt, cv::Mat img,
               const std::vector<std::vector<int>> &used);

std::vector<cv::Point> get_neighb(cv::Point pt,
                                  const std::vector<std::vector<int>> &used);

FindCompResult find_comp(cv::Mat img);

std::vector<cv::Point>
find_longest_path(const std::vector<std::vector<int>> &used, int idx,
                  cv::Point pt);

void dfs(cv::Point pt, std::vector<std::vector<int>> &used,
         std::vector<std::vector<int>> &nums,
         std::vector<std::vector<int>> &used_pok,
         std::vector<std::vector<cv::Point>> &cycle,
         std::vector<cv::Point> &st);

std::vector<std::vector<cv::Point>>
find_cycle(std::vector<std::vector<int>> used, int idx, cv::Point pt);

cv::Point2d get_rotation_angle(cv::Point2d st, cv::Point2d lst,
                               cv::Point2d fin);

bool can_erase_pix(cv::Mat &skelet, cv::Point pt,
                   std::vector<cv::Point> neighb);

double calc_line_dist(cv::Point pt1, cv::Point pt2, const cv::Mat &raw);

void swap_procedure(std::vector<std::vector<cv::Point>> &comp,
                    std::vector<std::vector<int>> &used, int cnt);

cv::Mat postproc_skelet(cv::Mat skelet, int threshold);

cv::Mat skeletonize(cv::Mat bin_mask);

std::pair<int, int> calc_img_sz(std::string path);

std::vector<int> create_partition(int max_num, int step = 100);

std::pair<cv::Mat, cv::Mat> downscale_recs(double downscale_factor, std::string path, std::string scroll_id, cv::Mat img1, cv::Mat img2);

std::pair<int, int> read_json(std::string json_path);

void clear_nans(std::string path, std::string scroll_id, int st_idx,
                int fin_idx, std::vector<int> nums);

uint8_t find_quantile_png(cv::Mat img, double quant);

cv::Mat autocontrast(cv::Mat img);

cv::Mat correct_orientation(cv::Mat img, bool hori_flip, bool vert_flip,
                            bool swap);

ConfigParams read_config(std::string path);

std::vector<int> read_nums(std::string path);

void init_filesystem(std::string cfg_path, ConfigParams params);

std::vector<std::string> get_dir_paths(std::string dir);

