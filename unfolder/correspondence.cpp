#include "correspondence.h"

//void build_rec_correspondence_from_scan(std::string scan_pt_3d_path,
//                                        std::string line_pts_full_path,
//                                        std::string out_rec_pt_3d_path,
//                                        int st_num, int fin_num, int step,
//                                        double downscale) {
void build_rec_correspondence_from_scan(std::string scan_pt_3d_path, std::string final_path, std::string details_path,
                                      std::string scroll_id,
                                      int st_idx, int fin_idx, double downscale, std::vector<int> nums, int slice_count) {
  std::ifstream in(scan_pt_3d_path);
  cv::Point pt;
  std::vector<cv::Point> vec_pt;
  std::vector<cv::Point3d> vec_pt_3d;
  while (in >> pt.x >> pt.y) {
    vec_pt.push_back(pt);
    cv::Point3d pt3;
    in >> pt3.x >> pt3.y >> pt3.z;
    vec_pt_3d.push_back(pt3);
  }

  std::vector<int> pt_idx_row(vec_pt.size()), pt_idx_col(vec_pt.size());
  std::vector<double> mn_dst(vec_pt.size(), 1000 * 1000 * 1000);
  std::vector<cv::Point3d> near_pts(vec_pt.size());

  for (int i = st_idx; i <= fin_idx; ++i) {
    std::cout << "Row " << i << std::endl;
    std::ifstream line_pts_prev(std::filesystem::path(details_path) / (scroll_id + "." + 
                                get_num(nums[i]) +  + ".line_pts_new.txt"));
    std::ifstream line_pts_next(std::filesystem::path(details_path) / (scroll_id + "." +
                                get_num(nums[std::min(i + 1, int(nums.size()) - 1)]) + ".line_pts_new.txt"));

    int col, col_next = -1;
    double x, y, x_next, y_next;

    int it = 0;

    while (line_pts_prev >> col >> x >> y) {
      if (i != fin_idx) {
        while (col_next < col) {
          line_pts_next >> col_next >> x_next >> y_next;
        }
        assert(col == col_next);
        for (int j = nums[i]; j <= nums[i + 1] - 1; ++j) {
          cv::Point3d pt_3d_near{x + (x_next - x) / (nums[i + 1] - nums[i]) * (j - nums[i]),
                                 y + (y_next - y) / (nums[i + 1] - nums[i]) * (j - nums[i]), j};
          //std::cout << j << std::endl;
          for (int num = 0; num < vec_pt.size(); ++num) {
            auto pt3 = vec_pt_3d[num];
            if (cv::norm(pt_3d_near - pt3) < mn_dst[num]) {
              mn_dst[num] = cv::norm(pt_3d_near - pt3);
              pt_idx_row[num] = j;
              pt_idx_col[num] = col;
              near_pts[num] = pt_3d_near;
            }
          }
        }
      } else {
        for (int j = nums[i]; j <= slice_count; ++j) {
          cv::Point3d pt_3d_near{x, y, j};
          for (int num = 0; num < vec_pt.size(); ++num) {
            auto pt = vec_pt[num];
            auto pt3 = vec_pt_3d[num];
            if (cv::norm(pt_3d_near - pt3) < mn_dst[num]) {
              mn_dst[num] = cv::norm(pt_3d_near - pt3);
              pt_idx_row[num] = j;
              pt_idx_col[num] = col;
              near_pts[num] = pt_3d_near;
            }
          }
        }
      }
    }
  }
  std::ofstream out(std::filesystem::path(final_path) / (scroll_id + ".unfolding_3d.txt"));

  for (int i = 0; i < near_pts.size(); ++i) {
    std::cout << mn_dst[i] << std::endl;
    out << int(round(pt_idx_row[i] / downscale)) << ' '
        << int(round(pt_idx_col[i] / downscale)) << ' ' << near_pts[i].x
        << ' ' << near_pts[i].y << ' ' << near_pts[i].z << "\n";
  }
  out.flush();
}