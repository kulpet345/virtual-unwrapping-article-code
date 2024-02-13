#pragma once

#include "alignment.h"
#include "new_skelet_conn.h"

void spiral_rec_nearest(int st_idx, int fin_idx,
                        bool fst_side, std::string scroll_id,
                        std::string out_path_details, std::vector<int> nums) {
  std::cout << "Start alignment" << std::endl;
  int st_num = nums[st_idx];
  int fin_num = nums[fin_idx];


  cv::Mat img;
  if (fst_side) {
    img = cv::imread(std::filesystem::path(out_path_details) / (scroll_id + ".1." + get_num(st_num) + "_" +
                  get_num(fin_num) + ".nonalign.png"), cv::IMREAD_GRAYSCALE);
  } else {
    img = cv::imread(std::filesystem::path(out_path_details) / (scroll_id + ".2." + get_num(st_num) + "_" +
                  get_num(fin_num) + ".nonalign.png"), cv::IMREAD_GRAYSCALE);
  }
  int cnt = img.size[1];
  //std::cout << cnt << std::endl;

  std::vector<int> pt_num1 = create_partition(cnt);

  std::vector<std::vector<int>> num_idxs;

  int dist = 500;

  //int st_num = nums[st_idx];
  //int fin_num = nums[fin_idx];

  //auto img = cv::imread(path + side + "_img_" + std::to_string(st_num) + "_" +
  //                          std::to_string(fin_num) + ".png",
  //                      cv::IMREAD_GRAYSCALE);

  //int old_st_num = st_num;

  //if (st_num % step == 0) {
  //  ++st_num;
  //}
  cv::Mat img1;
  cv::cvtColor(img, img1, cv::COLOR_GRAY2RGB);
  cv::Mat img2 = cv::Mat::zeros(img.size[0], img.size[1], CV_8UC1);
  std::vector<std::vector<std::pair<int, double>>> opt_paths;

  //std::ifstream in1(path + "pred_pts" + std::to_string(st_num) + ".txt");
  std::ifstream in1(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(st_num) + "_resample.txt"));

  std::vector<cv::Point2d> prev_pts(cnt);
  for (int j = 0; j < cnt; ++j) {
    in1 >> prev_pts[j].x >> prev_pts[j].y;
  }

  std::ifstream in2(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(fin_num) + "_resample.txt"));


  //std::ifstream in2(path + "pred_pts" + std::to_string(fin_num) + ".txt");
  std::vector<cv::Point2d> last_pts(cnt);
  for (int j = 0; j < cnt; ++j) {
    in2 >> last_pts[j].x >> last_pts[j].y;
  }

  std::vector<std::vector<int>> opt_poses(fin_idx - st_idx + 1);
  std::vector<cv::Point2d> prev_opt_pts(pt_num1.size());
  std::cout << st_idx << ' ' << nums[st_idx] << ' ' << fin_idx << ' ' << nums[fin_idx] << std::endl;
  // for (int i = st_num + step - ((st_num - 1) % step); i < 2632; i += step) {
  //for (int i = st_num; i < fin_num + 1; i += step) {
  for (int i = st_idx; i <= fin_idx; ++i) {
    std::cout << "Starting " << i << std::endl;
    int num = nums[i];
    //std::ifstream in3(path + "pred_pts" + std::to_string(i) + ".txt");
    std::ifstream in3(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(num) + "_resample.txt"));

    std::vector<cv::Point2d> cur_pts(cnt);
    for (int j = 0; j < cnt; ++j) {
      in3 >> cur_pts[j].x >> cur_pts[j].y;
    }
    int prev_opt = 0;
    for (int p = 0; p < pt_num1.size(); ++p) {
      std::cout << i << ' ' << p << std::endl;
      double mn_dist = 1000 * 1000 * 1000;
      int id = pt_num1[p];
      cv::Point2d norm_pt(
          prev_pts[id].x + (last_pts[id].x - prev_pts[id].x) * (nums[i] - st_num) /
                               (fin_num - st_num),
          prev_pts[id].y + (last_pts[id].y - prev_pts[id].y) * (nums[i] - st_num) /
                               (fin_num - st_num));
      int opt_id = -1;
      for (int k = std::max(std::max(pt_num1[p] - dist, prev_opt), 0);
            k < std::min(pt_num1[p] + dist, int(cur_pts.size())); ++k) {

        if ((cv::norm(cur_pts[k] - norm_pt) < mn_dist) &&
            (num == st_num || cv::norm(cur_pts[k] - prev_opt_pts[p]) < 5)) {
          mn_dist = cv::norm(cur_pts[k] - norm_pt);
          opt_id = k;
        }
      }
      if (opt_id == -1) {
        for (int k = std::max(std::max(pt_num1[p] - dist, prev_opt), 0);
              k < std::min(pt_num1[p] + dist, int(cur_pts.size())); ++k) {
          if (cv::norm(cur_pts[k] - norm_pt) < mn_dist) {
            mn_dist = cv::norm(cur_pts[k] - norm_pt);
            opt_id = k;
          }
        }
      }
      std::cout << "!!!!!" << std::endl;
      prev_opt_pts[p] = cur_pts[opt_id];
      prev_opt = opt_id;
      opt_poses[i - st_idx].push_back(opt_id);
      std::cout << "!!!!!" << std::endl;
    }
  }
  for (int i = st_idx; i <= fin_idx; ++i) {
    for (int p = 0; p < pt_num1.size(); ++p) {
      img1.at<cv::Vec3b>(nums[i], opt_poses[i - st_idx][p]).val[0] = 255;
      img1.at<cv::Vec3b>(nums[i], opt_poses[i - st_idx][p]).val[1] = 0;
      img1.at<cv::Vec3b>(nums[i], opt_poses[i - st_idx][p]).val[2] = 0;
    }
  }
  std::vector<double> pos(cnt);


  std::ifstream in(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(st_num) + "_resample.txt"));
  std::ofstream out(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(st_num) + ".line_pts.txt"));


  //std::ifstream in(path + "pred_pts" + std::to_string(st_num) + ".txt");
  //std::ofstream out(path + "line_pts" + std::to_string(st_num) + ".txt");

  cv::Mat img_3d = cv::Mat::zeros(img.size[0], img.size[1], CV_32FC3);

  auto new_pos = pos;
  for (int i = 0; i < cnt; ++i) {
    pos[i] = i;
    cv::Point2d pt;
    in >> pt.x >> pt.y;
    out << i << ' ' << pt.x << ' ' << pt.y << "\n";
    //for (int t = -st_num; t < 10; ++t) {
    //  img2.at<uint8_t>(st_num + t, pos[i]) =
    //      img.at<uint8_t>(st_num + t, pos[i]);
    for (int t = 0; t < nums[st_idx + 1]; ++t) {
      img2.at<uint8_t>(t, pos[i]) =
          img.at<uint8_t>(t, pos[i]);
      img_3d.at<cv::Vec3f>(t, pos[i])[0] = static_cast<float>(pt.x);
      img_3d.at<cv::Vec3f>(t, pos[i])[1] = static_cast<float>(pt.y);
      img_3d.at<cv::Vec3f>(t, pos[i])[2] =
          static_cast<float>(t);
    }
  }
  out.flush();
  std::vector<int> prev_poses, poses(opt_poses[0].size());

  for (int j = 0; j < pt_num1.size(); ++j) {
    prev_poses.push_back(pt_num1[j]);
  }

  std::cout << "OPT POSES" << std::endl;

  std::cout << opt_poses.size() << ' ' << opt_poses[0].size() << std::endl;

  for (int i = 1; i < opt_poses.size(); ++i) {
    std::cout << "St " << i << std::endl;
    for (int j = 0; j < opt_poses[0].size(); ++j) {
      poses[j] = opt_poses[i][j];
    }
    for (int j = 0; j <= opt_poses[0].size(); ++j) {
      std::cout << i << ' ' << j << std::endl;
      int l, r, l_prev, r_prev;
      if (j == 0) {
        l = 0;
        r = poses[0];
        l_prev = 0;
        r_prev = prev_poses[0];
      } else {
        if (j == opt_poses[0].size()) {
          l = poses.back();
          r = cnt - 1;
          l_prev = prev_poses.back();
          r_prev = cnt - 1;
        } else {
          l = poses[j - 1];
          r = poses[j];
          l_prev = prev_poses[j - 1];
          r_prev = prev_poses[j];
        }
      }
      for (int k = l; k <= r; ++k) {
        new_pos[k] =
            pos[l_prev] + double(k - l) / (r - l) * (pos[r_prev] - pos[l_prev]);
        //for (int t = 0; t < nums[i + st_idx] - nums[i - 1 + st_idx]; ++t) {
        //  new_pos[k] = pos[l_prev] +
        //               double(k - l) / (r - l) * (pos[r_prev] - pos[l_prev]);
        //}
      }
    }
    prev_poses = poses;

    std::ifstream in(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(nums[st_idx + i]) + "_resample.txt"));
    std::ofstream out(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(nums[st_idx + i]) + ".line_pts.txt"));

    //std::ofstream out(path + "line_pts" + std::to_string(st_num + step * i) +
    //                  ".txt");
    //std::ifstream in(path + "pred_pts" + std::to_string(st_num + step * i) +
    //                 ".txt");
    std::vector<cv::Point2d> pts;
    cv::Point2d pt;
    while (in >> pt.x >> pt.y) {
      pts.push_back(pt);
    }
    int slice_count = img.size[0];
    for (int j = 0; j + 1 < new_pos.size(); ++j) {
      for (int cur = int(ceil(new_pos[j])); cur <= int(floor(new_pos[j + 1]));
           ++cur) {
        out << cur << ' '
            << (pts[j] + (pts[j + 1] - pts[j]) * (cur - new_pos[j]) /
                             (new_pos[j + 1] - new_pos[j]))
                   .x
            << ' '
            << (pts[j] + (pts[j + 1] - pts[j]) * (cur - new_pos[j]) /
                             (new_pos[j + 1] - new_pos[j]))
                   .y
            << "\n";
        int r_bound = 0;
        //int r_bound = nums[i + st_idx] - nums[i - 1 + st_idx];
        if (i + st_idx != fin_idx) {
          r_bound = nums[i + st_idx + 1] - nums[i + st_idx];
        } else {
          r_bound = slice_count - fin_num;
        }
        for (int t = 0;
             t < r_bound;
             //std::min(step, fin_num - (st_num + step * i + step) + step + 1);
             ++t) {
          img2.at<uint8_t>(nums[i + st_idx] + t, cur) = uint8_t(
              round((double(img.at<uint8_t>(nums[i + st_idx] + t, j + 1)) *
                         (cur - new_pos[j]) +
                     double(img.at<uint8_t>(nums[i + st_idx] + t, j)) *
                         (new_pos[j + 1] - cur)) /
                    (new_pos[j + 1] - new_pos[j])));
          img_3d.at<cv::Vec3f>(nums[i + st_idx] + t, cur)[0] =
              static_cast<float>(
                  (pts[j] + (pts[j + 1] - pts[j]) * (cur - new_pos[j]) /
                                (new_pos[j + 1] - new_pos[j]))
                      .x);
          img_3d.at<cv::Vec3f>(nums[i + st_idx] + t, cur)[1] =
              static_cast<float>(
                  (pts[j] + (pts[j + 1] - pts[j]) * (cur - new_pos[j]) /
                                (new_pos[j + 1] - new_pos[j]))
                      .y);
          img_3d.at<cv::Vec3f>(nums[i + st_idx] + t, cur)[2] =
              static_cast<float>(nums[i + st_idx] + t);
        }
        if (i == opt_poses.size() - 1) {
          for (int t = 0; t < slice_count - fin_num; ++t) {
            img2.at<uint8_t>(fin_num + t, cur) = uint8_t(
                round((double(img.at<uint8_t>(fin_num + t, j + 1)) *
                           (cur - new_pos[j]) +
                       double(img.at<uint8_t>(fin_num + t, j)) *
                           (new_pos[j + 1] - cur)) /
                      (new_pos[j + 1] - new_pos[j])));
            img_3d.at<cv::Vec3f>(fin_num + t, cur)[0] =
                static_cast<float>(
                    (pts[j] + (pts[j + 1] - pts[j]) * (cur - new_pos[j]) /
                                  (new_pos[j + 1] - new_pos[j]))
                        .x);
            img_3d.at<cv::Vec3f>(fin_num + t, cur)[1] =
                static_cast<float>(
                    (pts[j] + (pts[j + 1] - pts[j]) * (cur - new_pos[j]) /
                                  (new_pos[j + 1] - new_pos[j]))
                        .y);
            img_3d.at<cv::Vec3f>(fin_num + t, cur)[2] =
                static_cast<float>(fin_num + t);
          }
        }
      }
    }
    out.flush();
    pos = new_pos;
  }
  cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + ".unfolding_3D.tif"), img_3d);
  //cv::imwrite(path + "rec_3d_" + std::to_string(st_num) + "_" +
  //                std::to_string(fin_num) + ".tif",
  //            img_3d);

  //int cur_num = st_num;
  /*
  for (int i = 0; i < opt_poses.size(); ++i) {
    int cur_num = nums[st_idx + i];
    std::ofstream out(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(st_num) + "_resample.txt"));

    //std::ofstream out(path + "cur_num" + std::to_string(cur_num) + ".txt");

    std::ifstream in3(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(cur_num) + "_resample.txt"));

    //std::ifstream in3(path + "pred_pts" + std::to_string(cur_num) + ".txt");
    std::vector<cv::Point2d> cur_pts(cnt);
    for (int j = 0; j < cnt; ++j) {
      in3 >> cur_pts[j].x >> cur_pts[j].y;
    }
    std::cout << opt_poses[i][0] << std::endl;
    for (int j = 0; j < opt_poses[i].size(); ++j) {
      out << cur_pts[opt_poses[i][j]].x << ' ' << cur_pts[opt_poses[i][j]].y
          << "\n";
    }
    //cur_num += step;

    out.flush();
  }
  */
  if (fst_side) {
    cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + ".1.align.png"), img2);
  } else {
    cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + ".2.align.png"), img2);
  }

  //cv::imwrite(path + side + "_img_align_" + std::to_string(old_st_num) + "_" +
  //                std::to_string(fin_num) + ".png",
  //            img2);

  std::cout << "End alignment" << std::endl;
}
