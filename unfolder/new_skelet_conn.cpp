#include "new_skelet_conn.h"

void delete_skelet_borders(cv::Mat &skelet) {
  for (int i = 0; i < skelet.size[0]; ++i) {
    skelet.at<uint8_t>(i, 0) = 0;
    skelet.at<uint8_t>(i, skelet.size[1] - 1) = 0;
  }
  for (int i = 0; i < skelet.size[1]; ++i) {
    skelet.at<uint8_t>(0, i) = 0;
    skelet.at<uint8_t>(skelet.size[0] - 1, i) = 0;
  }
}

void delete_big_cycles(cv::Mat &skelet, FindCompResult &comp_res, bool begin) {
  for (int j = 1; j <= comp_res.cnt; ++j) {

    auto cycle1 = find_cycle(comp_res.used, j, comp_res.comps[j][0]);
    if (cycle1.empty()) {
      continue;
    }

    for (int k = 0; k < cycle1.size(); ++k) {
      if (cycle1[k].size() < scroll_const.big_cycle_begin && begin ||
          cycle1[k].size() < scroll_const.big_cycle_middle && !begin) {
        for (auto el : cycle1[k]) {
          skelet.at<uint8_t>(el.x, el.y) = 0;
        }
      }
    }
  }
}

void delete_branch_points(FindCompResult &comp_res, cv::Mat &skelet,
                          bool st_slice) {
  std::vector<cv::Point> vec_el;
  for (int j = 0; j < comp_res.used.size(); ++j) {
    for (int k = 0; k < comp_res.used[0].size(); ++k) {
      if (comp_res.used[j][k] > 0) {
        cv::Point pt;
        pt.x = j;
        pt.y = k;
        if (get_neighb(pt, comp_res.used).size() >= 3) {
          if (st_slice) {
            vec_el.push_back(pt);
          } else {
            bool can = can_erase_pix(skelet, pt, get_neighb(pt, comp_res.used));
            if (can) {
              vec_el.push_back(pt);
              for (auto pt1 : get_neighb(pt, comp_res.used)) {
                vec_el.push_back(pt1);
                skelet.at<uint8_t>(pt1.x, pt1.y) = 0;
              }
              skelet.at<uint8_t>(pt.x, pt.y) = 0;
            }
          }
        }
      }
    }
  }
  for (auto el : vec_el) {
    comp_res.used[el.x][el.y] = 0;
  }
}

std::vector<std::vector<cv::Point>> collect_paths(FindCompResult &comp_res,
                                                  bool st_slice, bool debug) {
  std::vector<std::vector<cv::Point>> paths;
  for (int j = 1; j <= comp_res.cnt; ++j) {
    auto lol = find_longest_path(comp_res.used, j, comp_res.comps[j][0]);
    if (st_slice) {
      if (lol.size() > 3) {
        paths.push_back(lol);
      }
      continue;
    }

    auto cycle1 = find_cycle(comp_res.used, j, comp_res.comps[j][0]);
    if (cycle1.empty()) {
      if (lol.size() > 3) {
        paths.push_back(lol);
      }
      continue;
    }
    if (debug) {
      std::cout << "Find " << cycle1[0].size() << std::endl;
    }
    auto cycle = cycle1[0];

    for (int k = 0; k < cycle.size(); ++k) {
      auto el = cycle[k];
      if (get_neighb(el, comp_res.used).size() >= 3) {
        if (debug) {
          std::cout << "neighb" << std::endl;
        }
        for (auto el1 : cycle) {
          comp_res.used[el1.x][el1.y] = 0;
        }
        if (debug) {
          std::cout << el.x << ' ' << el.y << std::endl;
        }
        std::vector<cv::Point> new_cycle(cycle.size());
        std::copy(cycle.begin() + k, cycle.end(), new_cycle.begin());
        std::copy(cycle.begin(), cycle.begin() + k,
                  new_cycle.begin() + cycle.size() - k);
        cycle = new_cycle;
        cv::Point st_pt;
        for (auto el : comp_res.comps[j]) {
          if (comp_res.used[el.x][el.y]) {
            st_pt = el;
          }
        }
        auto path = find_longest_path(comp_res.used, j, st_pt);
        if (cv::norm(path.back() - el) < cv::norm(path[0] - el)) {
          std::reverse(path.begin(), path.end());
        }
        const int dif_len = scroll_const.curvature_dif_len;
        auto gt1 = el - path[dif_len];
        auto gt2 = path[dif_len] - path[2 * dif_len];
        auto pred1 = cycle[dif_len] - el;
        auto pred2 = cycle[cycle.size() - dif_len] - el;
        auto angle1 = get_rotation_angle(path[2 * dif_len], path[dif_len], el);
        auto angle2 = get_rotation_angle(path[dif_len], el, cycle[dif_len]);
        auto angle3 = get_rotation_angle(path[dif_len], el,
                                         cycle[cycle.size() - dif_len]);
        if (cv::norm(angle1 - angle3) > cv::norm(angle1 - angle2)) {
          cycle.pop_back();
          cycle.pop_back();
        } else {
          std::reverse(cycle.begin(), cycle.end());
          auto el = cycle.back();
          cycle.pop_back();
          cycle.pop_back();
          cycle.pop_back();
          cycle.push_back(el);
          std::reverse(cycle.begin(), cycle.end());
        }
        break;
      }
    }
    for (auto el : cycle) {
      comp_res.used[el.x][el.y] = j;
    }
    cv::Point st_pt;
    for (auto el : comp_res.comps[j]) {
      if (comp_res.used[el.x][el.y]) {
        st_pt = el;
      }
    }
    paths.push_back(find_longest_path(comp_res.used, j, st_pt));
  }
  return paths;
}

PathPos find_st_path_st_slice_cand(std::vector<std::vector<cv::Point>> &paths) {
  int len = 0;
  for (int i = 0; i < paths.size(); ++i) {
    len += paths[i].size();
  }
  double mx = 0;
  int st_idx, st_pos;
  for (int i = 0; i < paths.size(); ++i) {
    double mn = 100000;
    if (paths[i].size() < len * 0.001) {
      continue;
    }
    for (int j = 0; j < paths.size(); ++j) {
      if (i != j) {
        mn = std::min(cv::norm(paths[i][0] - paths[j][0]), mn);
        mn = std::min(cv::norm(paths[i][0] - paths[j].back()), mn);
      }
    }
    if (mn > mx) {
      mx = mn;
      st_idx = i;
      st_pos = 0;
    }
    mn = 100000;
    for (int j = 0; j < paths.size(); ++j) {
      if (i != j) {
        mn = std::min(cv::norm(paths[i].back() - paths[j][0]), mn);
        mn = std::min(cv::norm(paths[i].back() - paths[j].back()), mn);
      }
    }
    if (mn > mx) {
      mx = mn;
      st_idx = i;
      st_pos = -1;
    }
  }
  if (st_pos == 0) {
    return PathPos{st_idx, PathOrder::direct};
  }
  return PathPos{st_idx, PathOrder::reverse};
}

PathPos find_st_path_mid_slice(std::vector<std::vector<cv::Point>> &paths,
                               std::vector<std::pair<double, int>> &dst_idx,
                               std::vector<cv::Point> &big_path) {
  int mn = 100000;
  PathPos fst;
  for (int j = 0; j < paths.size(); ++j) {
    if (cv::norm(paths[j][0] - big_path[0]) < mn &&
        paths[j].size() > scroll_const.long_path &&
        dst_idx[j * 3 + 1].second <
            scroll_const.len_est_beg_mul * paths[j].size() &&
        scroll_const.len_est_beg_mul * dst_idx[j * 3 + 1].second >
            paths[j].size()) {
      if (paths[j].size() >= scroll_const.len_est_long_path) {
        if (dst_idx[j * 3 + 2].second <
            scroll_const.len_est_mid_mul * paths[j].size()) {
          mn = cv::norm(paths[j][0] - big_path[0]);
          fst.idx = j;
          fst.order = PathOrder::direct;
        }
      } else {
        mn = cv::norm(paths[j][0] - big_path[0]);
        fst.idx = j;
        fst.order = PathOrder::direct;
      }
    }
    if (cv::norm(paths[j].back() - big_path[0]) < mn &&
        paths[j].size() > scroll_const.long_path &&
        dst_idx[j * 3].second <
            scroll_const.len_est_beg_mul * paths[j].size() &&
        scroll_const.len_est_beg_mul * dst_idx[j * 3].second >
            paths[j].size()) {
      if (paths[j].size() >= scroll_const.len_est_long_path) {
        if (dst_idx[j * 3 + 2].second <
            scroll_const.len_est_mid_mul * paths[j].size()) {
          mn = cv::norm(paths[j].back() - big_path[0]);
          fst.idx = j;
          fst.order = PathOrder::reverse;
        }
      } else {
        mn = cv::norm(paths[j].back() - big_path[0]);
        fst.idx = j;
        fst.order = PathOrder::reverse;
      }
    }
  }
  return fst;
}

std::pair<bool, std::vector<PathPos>>
collect_nearest_paths_st_slice_cand(std::vector<cv::Point> &big_path,
                                    std::vector<std::vector<cv::Point>> &paths,
                                    std::vector<bool> &us_pth) {
  double mn = 100000;
  int idx = -1;
  int pos = 1;
  int len = 0;
  for (int i = 0; i < paths.size(); ++i) {
    len += paths[i].size();
  }
  for (int j = 0; j < us_pth.size(); ++j) {
    if (!us_pth[j]) {
      if (cv::norm(paths[j][0] - big_path.back()) < mn) {
        if (paths[j].size() >= len * 0.001) {
          mn = cv::norm(paths[j][0] - big_path.back());
          idx = j;
          pos = 0;
        }
      }
      if (cv::norm(paths[j].back() - big_path.back()) < mn) {
        if (paths[j].size() >= len * 0.001) {
          mn = cv::norm(paths[j].back() - big_path.back());
          idx = j;
          pos = -1;
        }
      }
    }
  }
  if (mn > scroll_const.long_dist) {
    return {true, {}};
  }
  std::vector<PathPos> idx_pos;
  for (int j = 0; j < us_pth.size(); ++j) {
    if (!us_pth[j]) {
      if (cv::norm(paths[j][0] - big_path.back()) <= scroll_const.long_dist) {
        mn = cv::norm(paths[j][0] - big_path.back());
        idx_pos.push_back(PathPos{j, PathOrder::direct});
      }
      if (cv::norm(paths[j].back() - big_path.back()) <=
          scroll_const.long_dist) {
        mn = cv::norm(paths[j].back() - big_path.back());
        idx_pos.push_back(PathPos{j, PathOrder::reverse});
      }
    }
  }
  return {false, idx_pos};
}

std::pair<bool, std::vector<PathPos>> collect_nearest_paths_mid_slice_cand(
    std::vector<cv::Point> &big_path,
    std::vector<std::vector<cv::Point>> &paths, std::vector<bool> &us_pth,
    std::vector<cv::Point> &pred_pts_raw, std::vector<cv::Point2d> &pred_pts,
    bool debug) {
  int len = 0;
  for (int i = 0; i < paths.size(); ++i) {
    len += paths[i].size();
  }
  double mn = 100000;
  int idx = -1;
  int pos = 1;
  for (int j = 0; j < us_pth.size(); ++j) {
    if (!us_pth[j]) {
      if (cv::norm(paths[j][0] - big_path.back()) < mn) {
        if (paths[j].size() >= 0.001 * len) {
          mn = cv::norm(paths[j][0] - big_path.back());
          idx = j;
          pos = 0;
        }
      }
      if (cv::norm(paths[j].back() - big_path.back()) < mn) {
        if (paths[j].size() >= 0.001 * len) {
          mn = cv::norm(paths[j].back() - big_path.back());
          idx = j;
          pos = -1;
        }
      }
    }
  }

  if (mn > scroll_const.long_dist &&
      cv::norm(cv::Point2d(big_path.back()) - pred_pts.back()) <
          scroll_const.near_to_fin) {
    return {true, {}};
  }
  if (abs(mn - 100000) < 1e-5) {
    return {true, {}};
  }

  std::vector<PathPos> idx_pos;
  for (int j = 0; j < us_pth.size(); ++j) {
    if (!us_pth[j]) {
      if (cv::norm(paths[j][0] - big_path.back()) <= scroll_const.long_dist ||
          (mn > scroll_const.long_dist &&
           abs(cv::norm(paths[j][0] - big_path.back()) - mn) < 1e-5)) {
        idx_pos.push_back({j, PathOrder::direct});
      }
      if (cv::norm(paths[j].back() - big_path.back()) <=
              scroll_const.long_dist ||
          (mn > scroll_const.long_dist &&
           abs(cv::norm(paths[j].back() - big_path.back()) - mn) < 1e-5)) {
        idx_pos.push_back({j, PathOrder::reverse});
      }
    }
  }
  return {false, idx_pos};
}

void st_slice_choose_best_cand(std::vector<PathPos> &idx_pos,
                               std::vector<std::vector<cv::Point>> &paths,
                               std::vector<cv::Point> &big_path, cv::Mat &raw,
                               std::vector<bool> &us_pth) {
  std::sort(idx_pos.begin(), idx_pos.end(), [&](PathPos p1, PathPos p2) {
    int idx1 = p1.idx;
    int idx2 = p2.idx;
    cv::Point end1, end2;
    if (p1.order == PathOrder::direct) {
      end1 = paths[idx1][0];
    } else {
      end1 = paths[idx1].back();
    }
    if (p2.order == PathOrder::direct) {
      end2 = paths[idx2][0];
    } else {
      end2 = paths[idx2].back();
    }

    return calc_line_dist(big_path.back(), end1, raw) <
           calc_line_dist(big_path.back(), end2, raw);
  });

  PathPos cur = idx_pos[0];

  int num = 0;
  while (true) {
    if (cur.order == PathOrder::direct) {
      if (cv::norm(paths[cur.idx][0] - big_path.back()) <=
          cv::norm(paths[cur.idx].back() - big_path.back())) {
        break;
      }
    } else {
      if (cv::norm(paths[cur.idx][0] - big_path.back()) >=
          cv::norm(paths[cur.idx].back() - big_path.back())) {
        break;
      }
    }
    ++num;
    if (num == idx_pos.size()) {
      break;
    }
    cur = idx_pos[num];
  }
  us_pth[cur.idx] = true;
  if (cur.order == PathOrder::reverse) {
    std::reverse(paths[cur.idx].begin(), paths[cur.idx].end());
  }
  big_path.resize(big_path.size() + paths[cur.idx].size());
  std::copy(paths[cur.idx].begin(), paths[cur.idx].end(),
            big_path.begin() + big_path.size() - paths[cur.idx].size());
}

bool CompSkeletsBackups::operator()(PathPos p1, PathPos p2) {
  int idx1 = p1.idx;
  int idx2 = p2.idx;
  cv::Point end1, end2;
  int val1, val2;
  double val3, val4;
  int val5, val6;
  int val7, val8;
  int val9, val10;
  double len1, len2;
  double val11, val12;

  double mn_dist_p1 = 1000000;
  int mn_idx_p1 = -1;
  if (p1.order == PathOrder::direct) {
    for (int t = 0; t < paths[idx1].size(); ++t) {
      if (cv::norm(cv::Point2d(paths[idx1][t]) - pred_pts[cur_min]) <
          mn_dist_p1) {
        mn_dist_p1 = cv::norm(cv::Point2d(paths[idx1][t]) - pred_pts[cur_min]);
        mn_idx_p1 = t;
      } else {
        break;
      }
    }
  } else {
    for (int t = paths[idx1].size() - 1; t >= 0; --t) {
      if (cv::norm(cv::Point2d(paths[idx1][t]) - pred_pts[cur_min]) <
          mn_dist_p1) {
        mn_dist_p1 = cv::norm(cv::Point2d(paths[idx1][t]) - pred_pts[cur_min]);
        mn_idx_p1 = t;
      } else {
        break;
      }
    }
  }

  double mn_dist_p2 = 1000000;
  int mn_idx_p2 = -1;
  if (p2.order == PathOrder::direct) {
    for (int t = 0; t < paths[idx2].size(); ++t) {
      if (cv::norm(cv::Point2d(paths[idx2][t]) - pred_pts[cur_min]) <
          mn_dist_p2) {
        mn_dist_p2 = cv::norm(cv::Point2d(paths[idx2][t]) - pred_pts[cur_min]);
        mn_idx_p2 = t;
      } else {
        break;
      }
    }
  } else {
    for (int t = paths[idx2].size() - 1; t >= 0; --t) {
      if (cv::norm(cv::Point2d(paths[idx2][t]) - pred_pts[cur_min]) <
          mn_dist_p2) {
        mn_dist_p2 = cv::norm(cv::Point2d(paths[idx2][t]) - pred_pts[cur_min]);
        mn_idx_p2 = t;
      } else {
        break;
      }
    }
  }

  if (p1.order == PathOrder::direct) {
    val1 = dst_idx[idx1 * 3].second;
    val3 = dst_idx[idx1 * 3 + 1].first;
    val5 = dst_idx[idx1 * 3 + 1].second;
    len1 =
        paths_dist[idx1][paths[idx1].size() - 1] - paths_dist[idx1][mn_idx_p1];
    val7 = dst_idx[idx1 * 3 + 2].second;
    val9 = dst_idx[idx1 * 3 + 1].second;
    val11 = dst_idx[idx1 * 3].first;
  } else {
    val1 = dst_idx[idx1 * 3 + 1].second;
    val3 = dst_idx[idx1 * 3].first;
    val5 = dst_idx[idx1 * 3].second;
    len1 = paths_dist[idx1][mn_idx_p1];
    val7 = dst_idx[idx1 * 3 + 2].second;
    val9 = dst_idx[idx1 * 3].second;
    val11 = dst_idx[idx1 * 3 + 1].first;
  }
  if (p2.order == PathOrder::direct) {
    val2 = dst_idx[idx2 * 3].second;
    val4 = dst_idx[idx2 * 3 + 1].first;
    val6 = dst_idx[idx2 * 3 + 1].second;
    len2 =
        paths_dist[idx2][paths[idx2].size() - 1] - paths_dist[idx2][mn_idx_p2];
    val8 = dst_idx[idx2 * 3 + 2].second;
    val10 = dst_idx[idx2 * 3 + 1].second;
    val12 = dst_idx[idx1 * 3].first;
  } else {
    val2 = dst_idx[idx2 * 3 + 1].second;
    val4 = dst_idx[idx2 * 3].first;
    val6 = dst_idx[idx2 * 3].second;
    len2 = paths_dist[idx2][mn_idx_p2];
    val8 = dst_idx[idx2 * 3 + 2].second;
    val10 = dst_idx[idx2 * 3].second;
    val12 = dst_idx[idx1 * 3 + 1].first;
  }

  if (abs(pref_dst[val5] - pref_dst[cur_min]) >= 5 * len1) {
    return false;
  }

  if (abs(pref_dst[val6] - pref_dst[cur_min]) >= 5 * len2) {
    return true;
  }
  if (val5 < cur_min) {
    return false;
  }
  if (val6 < cur_min) {
    return true;
  }
  if (paths[idx1].size() >= 20 && val7 > val1 && val7 > val9) {
    return false;
  }
  if (paths[idx2].size() >= 20 && val8 > val2 && val8 > val10) {
    return true;
  }
  if (paths[idx1].size() >= 20 && val7 < val1 && val7 > val9) {
    return false;
  }
  if (paths[idx2].size() >= 20 && val8 < val2 && val8 > val10) {
    return true;
  }
  if (val3 > 30) {
    return false;
  }
  if (val11 > 30) {
    return false;
  }
  if (val4 > 30) {
    return true;
  }
  if (val12 > 30) {
    return true;
  }
  int dif_pt = 20;
  if (paths[idx1].size() >= 100 && paths[idx2].size() < 100) {
    dif_pt = 30;
  }
  if (paths[idx2].size() >= 100 && paths[idx1].size() < 100) {
    dif_pt = 30;
  }
  if (abs(val1 - val2) > dif_pt) {
    return val1 < val2;
  }
  return val3 < val4;
}

void draw_result_st(cv::Mat &img, std::vector<cv::Point> &big_path,
                    std::string out_path, int st_num, std::string scroll_id) {
  cv::Mat path_im_raw = cv::Mat::zeros(img.size[0], img.size[1], CV_8UC1);
  for (int i = 0; i < big_path.size(); ++i) {
    path_im_raw.at<uint8_t>(big_path[i].x, big_path[i].y) =
        uint8_t(int(100 + double(i) / big_path.size() * 155));
  }
  cv::imwrite(std::filesystem::path(out_path) / (scroll_id + "." + get_num(st_num) + "_raw.png"),
            path_im_raw);
}

void write_pred_pts(std::vector<cv::Point2d> &pred_pts,
                    std::vector<cv::Point> &pred_pts_raw, int num,
                    std::string out_path, std::string scroll_id) {
  std::ofstream out1(std::filesystem::path(out_path) / (scroll_id + "." + get_num(num) + "_resample.txt"));

  for (int j = 0; j < pred_pts.size(); ++j) {
    out1 << pred_pts[j].x << ' ' << pred_pts[j].y << "\n";
  }

  out1.flush();

  std::ofstream out2(std::filesystem::path(out_path) / (scroll_id + "." + get_num(num) + "_raw.txt"));

  for (int j = 0; j < pred_pts_raw.size(); ++j) {
    out2 << pred_pts_raw[j].x << ' ' << pred_pts_raw[j].y << "\n";
  }

  out2.flush();
}

void texturing_operation(cv::Mat &fst_img, cv::Mat &sec_img,
                         int slice_num, std::vector<cv::Point2d> &fst_path,
                         bool debug, std::string raw_path,
                         std::string out_path, const std::set<int>& nums,
                         std::string scroll_id) {
  auto raw = read_tif(raw_path, 64);
  cv::Mat set_p = cv::Mat::zeros(raw.size[0], raw.size[1], CV_8UC1);
  cv::Mat set_q = cv::Mat::zeros(raw.size[0], raw.size[1], CV_8UC1);
  for (int i = 1; i < fst_path.size(); ++i) {
    cv::Point2d pt1 = fst_path[i];
    cv::Point2d pt2 = fst_path[i - 1];
    cv::Point2d dif_pt = pt1 - pt2;
    cv::Point2d norm_vec;
    norm_vec.x = -dif_pt.y;
    norm_vec.y = dif_pt.x;
    double nm = sqrt(dif_pt.x * dif_pt.x + dif_pt.y * dif_pt.y);
    cv::Point2d tot_norm = norm_vec / nm * scroll_const.norm_len;
    double val1 = 0.0, val2 = 0.0;
    cv::Point2d rt = pt1 + tot_norm;
    cv::Point l, r;
    l.x = int(round(pt1.x));
    l.y = int(round(pt1.y));
    r.x = int(round(rt.x));
    r.y = int(round(rt.y));
    cv::LineIterator it(l, r);
    for (int cnt = 0; cnt < it.count; ++cnt, ++it) {
      cv::Point qt;
      qt.x = it.pos().x;
      qt.y = it.pos().y;
      if (debug) {
        set_p.at<uint8_t>(qt.x, qt.y) = 255;
      }
      val1 = std::max(val1, raw.at<double>(qt.x, qt.y));
    }
    rt = pt1 - tot_norm;
    l.x = int(round(pt1.x));
    l.y = int(round(pt1.y));
    r.x = int(round(rt.x));
    r.y = int(round(rt.y));
    cv::LineIterator it1(l, r);
    for (int cnt = 0; cnt < it1.count; ++cnt, ++it1) {
      cv::Point qt;
      qt.x = it1.pos().x;
      qt.y = it1.pos().y;
      if (debug) {
        set_q.at<uint8_t>(qt.x, qt.y) = 255;
      }
      val2 = std::max(val2, raw.at<double>(qt.x, qt.y));
    }
    fst_img.at<double>(slice_num, i - 1) = val1;
    sec_img.at<double>(slice_num, i - 1) = val2;
  }
  if (nums.count(slice_num)) {
    if (debug) {
      cv::imwrite(std::filesystem::path(out_path) / (scroll_id + ".1." + get_num(slice_num) + "_norm.png"), set_p);
      cv::imwrite(std::filesystem::path(out_path) / (scroll_id + ".2." + get_num(slice_num) + "_norm.png"), set_q);
    }
  }
}


std::vector<std::vector<double>>
calc_pref_dist(std::vector<std::vector<cv::Point>> &paths) {
  std::vector<std::vector<double>> paths_dist(paths.size());
  for (int k = 0; k < paths.size(); ++k) {
    paths_dist[k].resize(paths[k].size());
    paths_dist[k][0] = 0;
    for (int j = 1; j < paths[k].size(); ++j) {
      paths_dist[k][j] =
          paths_dist[k][j - 1] + cv::norm(paths[k][j] - paths[k][j - 1]);
    }
  }
  return paths_dist;
}

std::vector<std::pair<double, int>>
calc_min_dist(std::vector<std::vector<cv::Point>> &paths,
              std::vector<cv::Point2d> &pred_pts) {
  std::vector<std::pair<double, int>> dst_idx;
  for (int k = 0; k < paths.size(); ++k) {
    double mn_dst1 = 100000;
    double mn_dst2 = 100000;
    double mn_dst3 = 100000;
    int mn_idx1 = -1;
    int mn_idx2 = -1;
    int mn_idx3 = -1;
    for (int j = 0; j < pred_pts.size(); ++j) {
      std::vector<double> beg_vec, end_vec, mid_vec;
      for (int t = 0; t < std::min(5, int(paths[k].size())); ++t) {
        beg_vec.push_back(
            cv::norm(cv::Point2d(paths[k][t].x, paths[k][t].y) - pred_pts[j]));
        end_vec.push_back(
            cv::norm(cv::Point2d(paths[k][paths[k].size() - 1 - t].x,
                                 paths[k][paths[k].size() - 1 - t].y) -
                     pred_pts[j]));
        if (paths[k].size() >= 100) {
          beg_vec.back() =
              cv::norm(cv::Point2d(paths[k][20 + t].x, paths[k][20 + t].y) -
                       pred_pts[j]);
          end_vec.back() =
              cv::norm(cv::Point2d(paths[k][paths[k].size() - 21 - t].x,
                                   paths[k][paths[k].size() - 21 - t].y) -
                       pred_pts[j]);
        }
        if (paths[k].size() >= 20) {
          mid_vec.push_back(
              cv::norm(cv::Point2d(paths[k][paths[k].size() / 2 + t].x,
                                   paths[k][paths[k].size() / 2 + t].y) -
                       pred_pts[j]));
        }
      }

      std::sort(beg_vec.begin(), beg_vec.end());
      std::sort(end_vec.begin(), end_vec.end());
      std::sort(mid_vec.begin(), mid_vec.end());

      if (beg_vec[beg_vec.size() / 2] < mn_dst1) {
        mn_dst1 = beg_vec[beg_vec.size() / 2];
        mn_idx1 = j;
      }
      if (end_vec[end_vec.size() / 2] < mn_dst2) {
        mn_dst2 = end_vec[end_vec.size() / 2];
        mn_idx2 = j;
      }

      if (paths[k].size() >= 20 && mid_vec[mid_vec.size() / 2] < mn_dst3) {
        mn_dst3 = mid_vec[mid_vec.size() / 2];
        mn_idx3 = j;
      }
    }
    dst_idx.push_back({mn_dst1, mn_idx1});
    dst_idx.push_back({mn_dst2, mn_idx2});
    dst_idx.push_back({mn_dst3, mn_idx3});
  }
  return dst_idx;
}

int calc_twisting(std::vector<cv::Point2d> &pred_pts) {
  int cnt_conn = 0;
  int prev_idx_conn = pred_pts.size();
  for (int t = pred_pts.size() - scroll_const.twisting_dif_len; t >= 0; --t) {
    if (prev_idx_conn >= t + scroll_const.twisting_dif_len &&
        cv::norm(pred_pts[t] - pred_pts.back()) <= scroll_const.twisting_dist) {
      ++cnt_conn;
      prev_idx_conn = t;
    }
  }
  return cnt_conn;
}

bool case_mid_nearest(std::vector<std::vector<cv::Point>> &paths,
                      std::map<cv::Point, int, CompClass> &map_pts,
                      double &tot_dist, int &cur_min,
                      std::vector<cv::Point> &big_path, PathOrder pos, int idx,
                      std::vector<std::vector<double>> paths_dist) {
  int mx_idx = -1;
  int best_pos = -1;
  for (int nxt_pos = 0; nxt_pos < paths[idx].size(); ++nxt_pos) {
    int nearest_idx = -1;
    int mn_dist = 1000 * 1000 * 1000;
    for (int k = -5; k <= 5; ++k) {
      for (int t = -5; t <= 5; ++t) {
        if (map_pts.count(cv::Point(paths[idx][nxt_pos].x + k,
                                    paths[idx][nxt_pos].y + t))) {
          if (k * k + t * t < mn_dist) {
            mn_dist = k * k + t * t;
            nearest_idx = map_pts[cv::Point(paths[idx][nxt_pos].x + k,
                                            paths[idx][nxt_pos].y + t)];
          }
        }
      }
    }
    if (nearest_idx > mx_idx) {
      mx_idx = nearest_idx;
      best_pos = nxt_pos;
    }
  }
  if (best_pos == -1) {
    std::cout << "BEST POS -1" << std::endl;
    return true;
  }
  assert(best_pos != -1);
  if (pos == PathOrder::direct) {
    tot_dist += cv::norm(big_path.back() - paths[idx][0]);
    tot_dist += paths_dist[idx][best_pos];
    big_path.resize(big_path.size() + best_pos + 1);
    std::copy(paths[idx].begin(), paths[idx].begin() + best_pos + 1,
              big_path.begin() + big_path.size() - best_pos - 1);
    cur_min = mx_idx;
    return true;
  } else {
    tot_dist += cv::norm(big_path.back() - paths[idx].back());
    tot_dist += paths_dist[idx].back() - paths_dist[idx][best_pos];
    big_path.resize(big_path.size() + paths[idx].size() - best_pos);
    std::reverse(paths[idx].begin(), paths[idx].end());
    best_pos = paths[idx].size() - best_pos - 1;
    std::copy(paths[idx].begin(), paths[idx].begin() + best_pos + 1,
              big_path.begin() + big_path.size() - best_pos - 1);
    cur_min = mx_idx;
    return false;
  }
}

void skelet_conn_new(std::vector<std::string> raw_paths, bool debug, bool swap_order,
                    std::string out_path_details, std::vector<std::string> mask_paths, std::vector<int> nums, int& st_idx, int fin_idx,
                    int slice_count, std::string scroll_id) {
  bool seg_st = false;
  int st_num, fin_num;
  int prev_write;
  int samples_count = 0;

  std::set<int> set_nums;
  for (int num: nums) {
    set_nums.insert(num);
  }

  std::vector<cv::Point2d> pred_pts;
  std::vector<cv::Point> pred_pts_raw;
  double pred_tot_dist;
  std::vector<cv::Point> big_path;




  while (!seg_st) {
    st_num = nums[st_idx];
    fin_num = nums[fin_idx];

    std::cout << "Skelet conn st " << st_num << std::endl;
    prev_write = -1;

    std::cout << "Start skelet conn algorithm" << std::endl;

    cv::Mat img = cv::imread(mask_paths[st_idx],
                   cv::IMREAD_GRAYSCALE);
    std::cout << "Succ read image" << std::endl;


  
  cv::Mat skelet = cv::Mat::zeros(img.size[0], img.size[1], CV_8UC1);
  cv::ximgproc::thinning((img > 0) * 255, skelet,
                         cv::ximgproc::THINNING_GUOHALL);
    if (debug) {
    cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(st_num) + ".skelet.png"),
                skelet);
  }
  delete_skelet_borders(skelet);
  auto comp_res = find_comp(skelet);

  if (swap_order) {
    swap_procedure(comp_res.comps, comp_res.used, comp_res.cnt);
  }

  if (debug) {
    std::cout << "Comps size" << comp_res.comps.size() << ' ' << comp_res.cnt
              << std::endl;
  }

  delete_big_cycles(skelet, comp_res, true);
  delete_branch_points(comp_res, skelet, true);


  if (debug) {
    cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(st_num) + ".skelet_last.png"),
            skelet);

  }
  comp_res = find_comp(skelet);

  if (swap_order) {
    swap_procedure(comp_res.comps, comp_res.used, comp_res.cnt);
  }

  auto paths = collect_paths(comp_res, true, debug);

  

  std::vector<bool> us_pth(paths.size());
  PathPos st_pos = find_st_path_st_slice_cand(paths);
  if (st_pos.order == PathOrder::reverse) {
    std::reverse(paths[st_pos.idx].begin(), paths[st_pos.idx].end());
  }
  us_pth[st_pos.idx] = true;
  big_path = paths[st_pos.idx];
  auto raw = image_to_int(
      read_tif(raw_paths[st_idx], 64));

    for (int i = 0; i < us_pth.size() - 1; ++i) {
      auto [finish, idx_pos] =
        collect_nearest_paths_st_slice_cand(big_path, paths, us_pth);

      if (finish) {
        break;
      }

    st_slice_choose_best_cand(idx_pos, paths, big_path, raw, us_pth);
  }

  int len_paths = 0;
  for (int i = 0; i < comp_res.cnt; ++i) {
    len_paths += comp_res.comps[i].size();
  }
  for (int i = 0; i < paths.size(); ++i) {
    std::cout << i << ' ' << paths[i].size() << ' ' << us_pth[i] << std::endl;
  }

  std::cout << "Len-plen " << big_path.size() << ' ' << len_paths << std::endl;

  if (big_path.size() >= 0.95 * len_paths) {
    seg_st = true;
  } else {
    ++st_idx;
    continue;
  }

  double dst = 0;
  double tot_dist = 0;
  for (int i = 1; i < big_path.size(); ++i) {
    dst += cv::norm(big_path[i] - big_path[i - 1]);
  }
  samples_count = int(trunc(dst));
  tot_dist = dst;

  if (debug) {
    draw_result_st(img, big_path, out_path_details, st_num, scroll_id);
  }

  std::vector<double> prev_idx;
  std::vector<cv::Point2d> prev_path;

  std::vector<cv::Point2d> fst_path;
  int prev_sz = 0;

  fst_path = uniform_sampling(big_path, samples_count);
  fst_path = gaus_smooth(fst_path);

  pred_pts = fst_path;
  pred_pts_raw = big_path;
  pred_tot_dist = tot_dist;

  write_pred_pts(fst_path, big_path, st_num, out_path_details, scroll_id);

  if (debug) {
  cv::Mat pred_pts_img =
    cv::Mat::zeros(skelet.size[0], skelet.size[1], CV_8UC1);
  for (int k = 0; k < fst_path.size(); ++k) {
    pred_pts_img.at<uint8_t>(fst_path[k].x, fst_path[k].y) = 255;
  }

  cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(st_num) + "_resample" + ".png"),
              pred_pts_img);
  }
  }


   
  auto fst_path = pred_pts;
  cv::Mat fst_img = cv::Mat::zeros(slice_count, samples_count, CV_64FC1);
  cv::Mat sec_img = cv::Mat::zeros(slice_count, samples_count, CV_64FC1);


  for (int k = st_num; k >= 0; --k) {
    std::cout << "Cur k" << ' ' << k << std::endl;
    texturing_operation(fst_img, sec_img,
                         k, fst_path,
                         debug, raw_paths[k],
                         out_path_details, set_nums, scroll_id);

    std::cout << "End texture" << std::endl;
  }
  for (int k = st_num; k < nums[st_idx + 1]; ++k) {
    std::cout << "Cur k" << ' ' << k << std::endl;
    texturing_operation(fst_img, sec_img,
                         k, fst_path,
                         debug, raw_paths[k],
                         out_path_details, set_nums, scroll_id);

    std::cout << "End texture" << std::endl;
  }

  cv::Mat fst_img_pok, sec_img_pok;






  for (int pok_idx = st_idx + 1; pok_idx <= fin_idx; ++pok_idx) {
    std::cout << "Cur raw " << nums[pok_idx] << std::endl;
    std::cout << "Start binary mask processing" << std::endl;
    int slice_num = nums[pok_idx];
    double tot_dist = 0;
    cv::Mat img;
    img = cv::imread(mask_paths[pok_idx],
                     cv::IMREAD_GRAYSCALE);

    std::map<cv::Point, int, CompClass> map_pts, map_pts_fixed;
    for (int j = 0; j < pred_pts.size(); ++j) {
      map_pts[cv::Point(int(round(pred_pts[j].x)), int(round(pred_pts[j].y)))] =
          j;
    }
    auto pred_pts_fixed = resample_fixed(pred_pts, 1);
    for (int j = 0; j < pred_pts_fixed.size(); ++j) {
      map_pts_fixed[cv::Point(int(round(pred_pts_fixed[j].x)),
                              int(round(pred_pts_fixed[j].y)))] = j;
    }
    cv::Mat skelet = cv::Mat::zeros(img.size[0], img.size[1], CV_8UC1);
    cv::ximgproc::thinning((img > 0) * 255, skelet,
                           cv::ximgproc::THINNING_GUOHALL);

    delete_skelet_borders(skelet);
    if (debug) {
      cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(slice_num) + ".skelet.png"),
          skelet);
    }
    auto comp_res = find_comp(skelet);

    if (swap_order) {
      swap_procedure(comp_res.comps, comp_res.used, comp_res.cnt);
    }
    if (debug) {
      std::cout << "Find comps" << std::endl;
    }
    std::vector<std::vector<cv::Point>> paths;

    delete_big_cycles(skelet, comp_res, false);

    delete_branch_points(comp_res, skelet, false);

    comp_res = find_comp(skelet);
    if (swap_order) {
      swap_procedure(comp_res.comps, comp_res.used, comp_res.cnt);
    }

    paths = collect_paths(comp_res, false, debug);

    std::vector<std::vector<double>> paths_dist = calc_pref_dist(paths);

    std::cout << "Finish binary mask processing" << std::endl;
    
    if (debug) {
      cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(slice_num) + ".skelet_last.png"),
            skelet);
    }

    std::cout << "Start skelet conn reconstruction" << std::endl;

    std::vector<bool> us_pth(paths.size());

    std::vector<std::pair<double, int>> dst_idx =
        calc_min_dist(paths, pred_pts);

    std::vector<double> pref_dst(pred_pts.size());
    pref_dst[0] = 0;

    for (int k = 1; k < pred_pts.size(); ++k) {
      pref_dst[k] = pref_dst[k - 1] + cv::norm(pred_pts[k] - pred_pts[k - 1]);
    }

    PathPos fst = find_st_path_mid_slice(paths, dst_idx, big_path);

    std::vector<int> cur_min_pts(us_pth.size());

    cur_min_pts[0] = 0;
    if (debug) {
      std::cout << "FST IDX " << fst.idx << std::endl;
    }

    big_path = paths[fst.idx];
    tot_dist += paths_dist[fst.idx].back();
    if (fst.order == PathOrder::reverse) {
      std::reverse(big_path.begin(), big_path.end());
    }
    us_pth[fst.idx] = true;

    int cur_min = dst_idx[fst.idx * 3 + 1].second;
    if (fst.order == PathOrder::reverse) {
      cur_min = dst_idx[fst.idx * 3].second;
    }

    if (debug) {
      std::cout << "CUR MIN " << cur_min << ' ' << pred_pts.size() << std::endl;
    }

    int cnt_conn = calc_twisting(pred_pts);

    std::vector<int> big_path_poses(us_pth.size() + 1);
    big_path_poses[0] = 0;

    std::vector<int> pos_us_pth{fst.idx};

    bool can_finish = false;
    bool fst_it = true;
    int backup_idx = -1;
    int backup_pos = -1;

    int fin_pos = -1;
    int cnt_raw = 0;
    bool empty_path = false;

    while (!can_finish) {

      CompSkeletsBackups comp_func{paths,    paths_dist, pred_pts,
                                   pref_dst, cur_min,    dst_idx};

      fin_pos = -1;
      int l_pos = 0;
      if (backup_idx != -1) {
        l_pos = backup_idx - 1;
      }

      for (int k = l_pos; k < us_pth.size() - 1; ++k) {
        fin_pos = k;
        big_path_poses[k + 1] = big_path.size();

        if (cv::norm(cv::Point2d(big_path.back().x, big_path.back().y) -
                     pred_pts.back()) <= scroll_const.twisting_dist) {
          if (debug) {
            std::cout << "CNT CONN " << cnt_conn << std::endl;
          }
          if (cnt_conn == 0) {
            break;
          }
          --cnt_conn;
        }
        auto [fin, idx_pos] = collect_nearest_paths_mid_slice_cand(
            big_path, paths, us_pth, pred_pts_raw, pred_pts, debug);
        if (debug) {
          std::cout << "Current seg " << k << std::endl;
          std::cout << big_path.back().x << ' ' << big_path.back().y
                    << std::endl;
        }

        if (debug) {
          for (int t = 0; t < idx_pos.size(); ++t) {
            std::cout << paths[idx_pos[t].idx][0].x << ' '
                      << paths[idx_pos[t].idx][0].y << ' '
                      << paths[idx_pos[t].idx].back().x << ' '
                      << paths[idx_pos[t].idx].back().y << std::endl;
          }
        }
        cur_min_pts[k + 1] = cur_min;
        comp_func.cur_min = cur_min;
        if (fin) {
          fin_pos = k;
          break;
        }
        if (debug) {
          std::cout << "idx pos " << idx_pos.size() << ' ' << idx_pos[0].idx
                    << ' ' << idx_pos[0].order << std::endl;
        }

        std::sort(idx_pos.begin(), idx_pos.end(), comp_func);
        int idx = idx_pos[0].idx;
        PathOrder pos = idx_pos[0].order;
        if (!fst_it && backup_idx == k + 1) {
          if (backup_pos == idx_pos.size()) {
            if (debug) {
              std::cout << "BACKUP MAX SIZE " << backup_idx << ' ' << backup_pos
                        << std::endl;
            }
            can_finish = true;
            break;
          }
          idx = idx_pos[backup_pos].idx;
          pos = idx_pos[backup_pos].order;
        }
        int num = 0;
        if (num == idx_pos.size()) {
          fin_pos = k;
          break;
        }

        us_pth[idx] = true;
        pos_us_pth.push_back(idx);
        bool reversed = false;
        if (pos == PathOrder::reverse) {
          reversed = true;
          std::reverse(paths[idx].begin(), paths[idx].end());
        }

        double mn_dist_beg = 1000000;
        int mn_idx_beg = -1;
        for (int t = 0; t < paths[idx].size(); ++t) {
          if (cv::norm(cv::Point2d(paths[idx][t]) - pred_pts[cur_min]) <
              mn_dist_beg) {
            mn_dist_beg =
                cv::norm(cv::Point2d(paths[idx][t]) - pred_pts[cur_min]);
            mn_idx_beg = t;
          } else {
            break;
          }
        }
        if (pos == PathOrder::direct) {
          cur_min = dst_idx[idx * 3 + 1].second;
        } else {
          cur_min = dst_idx[idx * 3].second;
        }

        cv::LineIterator it_fin(big_path.back(), paths[idx][mn_idx_beg]);

        double mn_norm = 1000 * 1000 * 1000;
        cv::Point mn_pt;
        for (int cnt = 0; cnt < it_fin.count; ++cnt, ++it_fin) {
          if (cv::norm(cv::Point2d(it_fin.pos()) - pred_pts.back()) < mn_norm) {
            mn_norm = cv::norm(cv::Point2d(it_fin.pos()) - pred_pts.back());
            mn_pt = it_fin.pos();
          }
        }

        if (debug) {    
         std::cout << "Can finish? "  << scroll_const.near_to_fin << std::endl;
        }

        if (mn_norm < 25 && big_path.size() >= 0.9 * pred_pts_raw.size() &&
            cnt_conn == 0) {
          if (debug) {
            std::cout << "OK FIN" << std::endl;
          }
          big_path.push_back(mn_pt);
          if (reversed) {
            std::reverse(paths[idx].begin(), paths[idx].end());
          }
          fin_pos = k;
          break;
        }

        cv::LineIterator it_fin1(big_path.back(), paths[idx][0]);

        mn_norm = 1000 * 1000 * 1000;
        for (int cnt = 0; cnt < it_fin1.count; ++cnt, ++it_fin1) {
          if (cv::norm(cv::Point2d(it_fin1.pos()) - pred_pts.back()) <
              mn_norm) {
            mn_norm = cv::norm(cv::Point2d(it_fin1.pos()) - pred_pts.back());
            mn_pt = it_fin1.pos();
          }
        }
        if (debug) {
          std::cout << "Can finish? " << mn_norm << ' '
                    << scroll_const.near_to_fin << std::endl;
        }

        if (mn_norm < 20 && big_path.size() >= 0.9 * pred_pts_raw.size() &&
            cnt_conn == 0) {
          if (debug) {
            std::cout << "OK FIN" << std::endl;
          }
          big_path.push_back(mn_pt);
          if (reversed) {
            std::reverse(paths[idx].begin(), paths[idx].end());
          }
          fin_pos = k;
          break;
        }
        tot_dist += cv::norm(big_path.back() - paths[idx][mn_idx_beg]);
        big_path.resize(big_path.size() + paths[idx].size() - mn_idx_beg);
        std::copy(paths[idx].begin() + mn_idx_beg, paths[idx].end(),
                  big_path.begin() + big_path.size() - paths[idx].size() +
                      mn_idx_beg);
        if (debug) {
          std::cout << "Cur min before " << mn_idx_beg << ' '
                    << paths[idx].size() << ' ' << cur_min << std::endl;
        }
        while (cur_min + 1 < pred_pts.size() &&
               cv::norm(cv::Point2d(big_path.back()) - pred_pts[cur_min + 1]) <
                   cv::norm(cv::Point2d(big_path.back()) - pred_pts[cur_min])) {
          ++cur_min;
        }

        if (debug) {
          std::cout << "Cur min after " << cur_min << std::endl;
        }
        bool end = false;
        for (int t = paths[idx].size() - 1; t >= 0; --t) {
          if (cv::norm(cv::Point2d(paths[idx][t]) - pred_pts.back()) < 20 &&
              (paths[idx].size() - t <= scroll_const.long_path_cont) &&
              big_path.size() - paths[idx].size() + std::max(t, mn_idx_beg) +
                      1 >=
                  0.9 * pred_pts_raw.size()) {
            if (debug) {
              std::cout << "Prev, cur raw paths sizes " << pred_pts_raw.size()
                        << ' ' << big_path.size() << std::endl;
            }
            for (int p = 0; p < paths[idx].size() - std::max(t, mn_idx_beg) - 1;
                 ++p) {
              big_path.pop_back();
            }
            end = true;
            tot_dist += paths_dist[idx][paths_dist[idx].size() - 1 -
                                        std::max(t, mn_idx_beg)] -
                        paths_dist[idx][mn_idx_beg];
            break;
          }
        }
        if (!end) {
          tot_dist += paths_dist[idx].back() - paths_dist[idx][mn_idx_beg];
        }
        bool cant_end = false;
        for (int t = 0; t < us_pth.size(); ++t) {
          if (!us_pth[t] && paths[t].size() >= scroll_const.long_path_cont) {
            cant_end = true;
          }
        }
        if (reversed) {
          std::reverse(paths[idx].begin(), paths[idx].end());
        }
        if (end && !cant_end) {
          fin_pos = k;
          break;
        }
      }
      if (debug) {
        cv::Mat path_im_raw = cv::Mat::zeros(img.size[0],
                                             img.size[1], CV_8UC1);
        for (int j = 0; j < big_path.size(); ++j) {
          path_im_raw.at<uint8_t>(big_path[j].x,
                                  big_path[j].y) =
              uint8_t(int(100 + double(j) / big_path.size() * 155));
        }
        if (debug) {
          cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(slice_num) + "_raw.png"),
                      path_im_raw);
        }
      }

      std::cout << "Pred size, post size " << pred_pts_raw.size() << ' '
                << big_path.size() << std::endl;

      if (big_path.empty() || can_finish) {
        std::cout << "Bad slice segmentation, try previous" << std::endl;
        fst_path = pred_pts;
        big_path = pred_pts_raw;
        empty_path = true;
        break;
      }

      fst_path = uniform_sampling(big_path, samples_count);
      fst_path = gaus_smooth(fst_path);

      std::vector<double> pref_dst_cur(fst_path.size()),
          pref_dst_prev(pred_pts.size());

      double cur_dst = 0;
      for (int j = 1; j < fst_path.size(); ++j) {
        pref_dst_cur[j] +=
            pref_dst_cur[j - 1] + cv::norm(fst_path[j] - fst_path[j - 1]);
        cur_dst += cv::norm(fst_path[j] - fst_path[j - 1]);
      }
      double prev_dst = 0;
      for (int j = 1; j < pred_pts.size(); ++j) {
        pref_dst_prev[j] +=
            pref_dst_prev[j - 1] + cv::norm(pred_pts[j] - pred_pts[j - 1]);
        prev_dst += cv::norm(pred_pts[j] - pred_pts[j - 1]);
      }
      if (can_finish) {
        break;
      }

      int prev_backup_pos = backup_pos;
      int prev_backup_idx = backup_idx;
      auto fst_path_fixed = resample_fixed(fst_path, 1);
      std::cout << "In backup loop" << std::endl;
      can_finish = false;
      fst_it = false;
      int cnt_disconn = 0;
      int max_cnt_disconn = 0;
      cv::Mat pos_img = cv::Mat::zeros(img.size[0], img.size[1], CV_8UC1);
      for (int t = 0; t < big_path.size(); ++t) {
        auto el = big_path[t];
        pos_img.at<uint8_t>(el.x, el.y) = 100 + 150 * t / big_path.size();
      }
      if (debug) {
        cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(slice_num) + "." + std::to_string(cnt_raw) + "_backup_raw.png"),
                    pos_img);
      }
      ++cnt_raw;

      for (int t = 0; t < fst_path_fixed.size(); ++t) {
        int opt_id = -1;
        int mn_dst = 1000 * 1000 * 1000;
        for (int a = -10; a <= 10; ++a) {
          for (int b = -10; b <= 10; ++b) {
            if (map_pts_fixed.count(
                    cv::Point(int(round(fst_path_fixed[t].x + a)),
                              int(round(fst_path_fixed[t].y + b))))) {
              if (a * a + b * b < mn_dst) {
                mn_dst = a * a + b * b;
                opt_id = map_pts_fixed[cv::Point(
                    int(round(fst_path_fixed[t].x + a)),
                    int(round(fst_path_fixed[t].y + b)))];
              }
            }
          }
        }
        if (opt_id == -1 || abs(t - opt_id) > 0.01 * fst_path_fixed.size()) {
          ++cnt_disconn;
          max_cnt_disconn = std::max(cnt_disconn, max_cnt_disconn);
        } else {
          if (cnt_disconn >= fst_path_fixed.size() * 0.03) {
            if (debug) {
              std::cout << "Cnt disconn " << t << ' ' << fst_path.size()
                        << std::endl;
            }
            double big_p_dst = 1000 * 1000 * 1000;
            int num_big_p = -1;
            for (int u = 0; u < big_path.size(); ++u) {
              if (cv::norm(cv::Point2d(big_path[u].x, big_path[u].y) -
                           fst_path_fixed[t - cnt_disconn]) < big_p_dst &&
                  abs(double(u) / big_path.size() -
                      double(t - cnt_disconn) / fst_path_fixed.size()) < 0.05) {
                big_p_dst = cv::norm(cv::Point2d(big_path[u].x, big_path[u].y) -
                                     fst_path_fixed[t - cnt_disconn]);
                num_big_p = u;
              }
            }

            if (debug) {

              std::cout << big_path[num_big_p].x << ' ' << big_path[num_big_p].y
                        << std::endl;
            }
            int path_pos = 1;
            while (path_pos - 1 <= fin_pos &&
                   big_path_poses[path_pos] <= num_big_p) {
              ++path_pos;
            }
            --path_pos;

            if (debug) {
              std::cout << "Big path pos " << big_path_poses[path_pos] - 1
                        << ' ' << num_big_p << std::endl;
              std::cout << big_path[big_path_poses[path_pos + 1] - 1].x << ' '
                        << big_path[big_path_poses[path_pos + 1] - 1].y
                        << std::endl;
              std::cout << big_path[num_big_p].x << ' ' << big_path[num_big_p].y
                        << std::endl;
            }
            if (backup_idx == path_pos) {
              ++backup_pos;
            } else {
              backup_idx = path_pos;
              backup_pos = 1;
            }
            cnt_disconn = 0;
            break;
          }
          cnt_disconn = 0;
        }
      }
      std::cout << "Max sequence to backup, final sequence to backup, fixed "
                   "length path size: "
                << max_cnt_disconn << ' ' << cnt_disconn << ' '
                << fst_path_fixed.size() << std::endl;
      if (cnt_disconn >= fst_path_fixed.size() * 0.03) {
        if (debug) {
          std::cout << "End cnt disconn " << 40234924 << ' ' << fst_path.size()
                    << std::endl;
        }
        double big_p_dst = 1000 * 1000 * 1000;
        int num_big_p = -1;
        for (int u = 0; u < big_path.size(); ++u) {
          if (cv::norm(cv::Point2d(big_path[u].x, big_path[u].y) -
                       fst_path_fixed[fst_path_fixed.size() - cnt_disconn]) <
                  big_p_dst &&
              abs(double(u) / big_path.size() -
                  double(fst_path_fixed.size() - cnt_disconn) /
                      fst_path_fixed.size()) < 0.05) {
            big_p_dst =
                cv::norm(cv::Point2d(big_path[u].x, big_path[u].y) -
                         fst_path_fixed[fst_path_fixed.size() - cnt_disconn]);
            num_big_p = u;
          }
        }
        if (debug) {
          std::cout << "Num" << ' ' << num_big_p << ' ' << big_path[num_big_p].x
                    << ' ' << big_path[num_big_p].y << std::endl;
        }
        int path_pos = 1;
        while (path_pos - 1 <= fin_pos &&
               big_path_poses[path_pos] <= num_big_p) {
          ++path_pos;
        }
        --path_pos;
        if (big_path_poses[path_pos] < num_big_p &&
            big_path_poses[path_pos + 1] - 5 <= num_big_p) {
          ++path_pos;
        }
        if (backup_idx == path_pos) {
          ++backup_pos;
        } else {
          backup_idx = path_pos;
          backup_pos = 1;
        }
        std::cout << "BACKUP" << ' ' << backup_idx << ' ' << backup_pos
                  << std::endl;
        std::cout << big_path[big_path_poses[backup_idx]].x << ' '
                  << big_path[big_path_poses[backup_idx]].y << std::endl;
      }

      if (backup_pos == prev_backup_pos && backup_idx == prev_backup_idx) {
        break;
      }

      cv::Mat pos_img_small = cv::Mat::zeros(img.size[0], img.size[1], CV_8UC1);
      for (int t = 0; t < big_path_poses[backup_idx]; ++t) {
        auto el = big_path[t];
        pos_img_small.at<uint8_t>(el.x, el.y) = 100 + 150 * t / big_path.size();
      }

      if (debug) {
        cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(slice_num) + "." + std::to_string(cnt_raw - 1) + "_backup.png"),
          pos_img);
      }

      if (backup_idx == -1) {
        if (debug) {
          std::cout << "BACKUP" << ' ' << -1 << std::endl;
        }
        can_finish = true;
        fst_it = false;
        break;
      }

      big_path.resize(big_path_poses[backup_idx]);
      cur_min = cur_min_pts[backup_idx];

      if (debug) {
        std::cout << "Pre use" << std::endl;
        std::cout << big_path.back().x << ' ' << big_path.back().y << std::endl;

        for (auto el : pos_us_pth) {
          std::cout << el << ' ';
        }
        std::cout << std::endl;
      }

      while (pos_us_pth.size() >= backup_idx + 1) {
        us_pth[pos_us_pth.back()] = false;
        pos_us_pth.pop_back();
      }
      fst_it = false;
      continue;
    }

    if (!empty_path) {
      double mn_dst = 1000 * 1000 * 1000;
      int fin_idx = -1;
      big_path_poses[fin_pos + 2] = big_path.size();
      double prev_dst = 0;
      for (int j = 1; j < pred_pts.size(); ++j) {
        prev_dst += cv::norm(pred_pts[j] - pred_pts[j - 1]);
      }

      for (int t = -1; t <= fin_pos; ++t) {
        double cur_dst_pref = 0;
        std::vector<cv::Point> big_path_pref(big_path_poses[t + 2]);
        std::copy(big_path.begin(), big_path.begin() + big_path_poses[t + 2],
                  big_path_pref.begin());
        auto fst_path_pref = uniform_sampling(big_path_pref, samples_count);
        fst_path_pref = gaus_smooth(fst_path_pref);
        double cur_dst = 0;
        for (int j = 1; j < fst_path_pref.size(); ++j) {
          cur_dst += cv::norm(fst_path_pref[j] - fst_path_pref[j - 1]);
        }

        if (debug) {

          std::cout << big_path_poses[t + 2] << ' ' << big_path.size()
                    << std::endl;

          std::cout << "Try pos " << t << ' ' << cur_dst << ' ' << prev_dst
                    << ' ' << fin_pos << std::endl;
        }
        if (cv::norm(big_path[big_path_poses[t + 2] - 1] -
                     pred_pts_raw.back()) < mn_dst &&
            0.95 * cur_dst <= prev_dst && 0.95 * prev_dst <= cur_dst) {
          mn_dst = cv::norm(big_path[big_path_poses[t + 2] - 1] -
                            pred_pts_raw.back());
          fin_idx = t;
        }
      }

      if (debug) {
        std::cout << "FIN IDX " << fin_idx << std::endl;
      }

      int t = big_path_poses[fin_idx + 2];
      while (big_path.size() > t) {
        big_path.pop_back();
      }

      fst_path = uniform_sampling(big_path, samples_count);
      fst_path = gaus_smooth(fst_path);
      fst_it = false;
      can_finish = true;

      if (debug) {
        std::cout << "OLD PRED PTS " << pred_pts.size() << ' '
                  << pred_pts_raw.size() << std::endl;
      }

      double cur_dst = 0;
      prev_dst = 0;

      for (int t = 1; t < pred_pts.size(); ++t) {
        prev_dst += cv::norm(pred_pts[t] - pred_pts[t - 1]);
      }
      for (int t = 1; t < fst_path.size(); ++t) {
        cur_dst += cv::norm(fst_path[t] - fst_path[t - 1]);
      }

      if (cur_dst >= 0.95 * prev_dst) {
        std::cout << "Finish skelet conn reconstruction" << std::endl;
        pred_pts = fst_path;
        pred_pts_raw = big_path;
        pred_tot_dist = tot_dist;
      } else {
        if (debug) {
          std::cout << "Slice" << std::endl;
        }
        std::cout << "Bad slice segmentation, try previous" << std::endl;
        big_path = pred_pts_raw;
        fst_path = pred_pts;
        break;
      }
      if (debug) {
        std::cout << "Pre break" << std::endl;
      }
    }

    if (debug) {
      std::cout << "NEXT PRED PTS " << fst_path.size() << ' ' << big_path.size()
                << std::endl;
    }
    write_pred_pts(pred_pts, pred_pts_raw, slice_num, out_path_details, scroll_id);

    if (debug) {
      std::cout << "Start texturing " << slice_num << std::endl;
    }

    int r_text = 0;
    if (slice_num == fin_num) {
      r_text = slice_count;
    } else {
      r_text = nums[pok_idx + 1];
    }

    if (debug) {
    cv::Mat pred_pts_img =
        cv::Mat::zeros(skelet.size[0], skelet.size[1], CV_8UC1);
    for (int k = 0; k < pred_pts.size(); ++k) {
      pred_pts_img.at<uint8_t>(pred_pts[k].x, pred_pts[k].y) = 255;
    }

    cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + "." + get_num(slice_num) + "_resample" + ".png"),
                pred_pts_img);
    }

    for (int w = slice_num; w < r_text; ++w) {
      if (debug) {
        std::cout << w << std::endl;
      }
      texturing_operation(fst_img, sec_img, w, fst_path, debug,
                          raw_paths[w], out_path_details, set_nums, scroll_id);
      std::cout << "Post texture " << w << std::endl;
    }
    if (slice_num / 40 != prev_write) {
      prev_write = slice_num / 40;
      if (debug) {
        cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + ".1." + get_num(slice_num) + "_nonalign.png"),
                  image_to_int(fst_img));
        cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + ".2." + get_num(slice_num) + "_nonalign.png"),
                  image_to_int(sec_img));
      }
    }
    if (debug) {
      std::cout << "Post texture" << std::endl;
    }
  }

  cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + ".1." + get_num(st_num) + "_" +
                  get_num(fin_num) + ".nonalign.tif"),
              fst_img);
  cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + ".2." + get_num(st_num) + "_" +
                  get_num(fin_num) + ".nonalign.tif"),
              sec_img);
  cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + ".1." + get_num(st_num) + "_" +
                  get_num(fin_num) + ".nonalign.png"),
              image_to_int(fst_img));
  cv::imwrite(std::filesystem::path(out_path_details) / (scroll_id + ".2." + get_num(st_num) + "_" +
                  get_num(fin_num) + ".nonalign.png"),
              image_to_int(sec_img));
}
