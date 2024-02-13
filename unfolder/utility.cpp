#include "utility.h"

std::string get_num(int n, int k) {
  std::string num = std::to_string(n);
  num = std::string(k - num.size(), '0') + num;
  return num;
}

std::vector<cv::Point2d> gaus_smooth(std::vector<cv::Point2d> pts) {
  cv::Mat kernel = cv::getGaussianKernel(7, 1);
  cv::Mat imgx = cv::Mat::zeros(pts.size(), 1, CV_64FC1);
  cv::Mat imgy = cv::Mat::zeros(pts.size(), 1, CV_64FC1);
  for (int i = 0; i < pts.size(); ++i) {
    imgx.at<double>(i, 0) = pts[i].x;
    imgy.at<double>(i, 0) = pts[i].y;
  }
  cv::sepFilter2D(imgx, imgx, CV_64FC1, kernel, cv::Mat::ones(1, 1, CV_64FC1));
  cv::sepFilter2D(imgy, imgy, CV_64FC1, kernel, cv::Mat::ones(1, 1, CV_64FC1));
  for (int i = 0; i < pts.size(); ++i) {
    pts[i] = cv::Point2d(imgx.at<double>(i, 0), imgy.at<double>(i, 0));
  }
  return pts;
}

void dfs_comp(int x, int y, const cv::Mat &img,
              std::vector<std::vector<int>> &out_comp, int cnt) {
  out_comp[x][y] = cnt;
  if (x >= 1 && img.at<uint8_t>(x - 1, y) > 0 && out_comp[x - 1][y] == 0) {
    dfs_comp(x - 1, y, img, out_comp, cnt);
  }
  if (y >= 1 && img.at<uint8_t>(x, y - 1) > 0 && out_comp[x][y - 1] == 0) {
    dfs_comp(x, y - 1, img, out_comp, cnt);
  }
  if (x + 1 < img.size[0] && img.at<uint8_t>(x + 1, y) > 0 &&
      out_comp[x + 1][y] == 0) {
    dfs_comp(x + 1, y, img, out_comp, cnt);
  }
  if (y + 1 < img.size[1] && img.at<uint8_t>(x, y + 1) > 0 &&
      out_comp[x][y + 1] == 0) {
    dfs_comp(x, y + 1, img, out_comp, cnt);
  }
}

std::vector<cv::Point2d> resample_fixed(std::vector<cv::Point2d> pts,
                                        double step) {
  std::vector<cv::Point2d> new_pts;
  new_pts.push_back(pts[0]);
  double curdl = 0;
  double curdr = cv::norm(pts[1] - pts[0]);
  int idx = 1;
  for (int i = 1;; ++i) {
    double sd = step * i;
    while (curdr < sd && idx + 1 < pts.size()) {
      curdl = curdr;
      curdr += cv::norm(pts[idx + 1] - pts[idx]);
      ++idx;
    }
    double dif = sd - curdl;
    if (dif > cv::norm(pts[idx] - pts[idx - 1])) {
      break;
    }
    cv::Point2d dbs;
    dbs.x = (pts[idx - 1].x +
             dif / (curdr - curdl) * (pts[idx].x - pts[idx - 1].x));
    dbs.y = (pts[idx - 1].y +
             dif / (curdr - curdl) * (pts[idx].y - pts[idx - 1].y));
    new_pts.push_back(dbs);
  }
  return new_pts;
}

double find_median(cv::Mat input) {
  input = input.reshape(0, 1);
  std::vector<double> vecFromMat;
  input.copyTo(vecFromMat);
  std::nth_element(vecFromMat.begin(),
                   vecFromMat.begin() + vecFromMat.size() / 2,
                   vecFromMat.end());
  double el1 = vecFromMat[vecFromMat.size() / 2];
  std::nth_element(vecFromMat.begin(),
                   vecFromMat.begin() + vecFromMat.size() / 2 - 1,
                   vecFromMat.end());
  double el2 = vecFromMat[vecFromMat.size() / 2 - 1];
  if (vecFromMat.size() % 2 == 1) {
    return el1;
  }
  return (el1 + el2) / 2;
}

double find_quantile(cv::Mat input, double quantile) {
  input = input.reshape(0, 1);
  std::vector<double> vecFromMat;
  input.copyTo(vecFromMat);
  std::nth_element(vecFromMat.begin(),
                   vecFromMat.begin() +
                       static_cast<int>(vecFromMat.size() * quantile),
                   vecFromMat.end());
  return vecFromMat[static_cast<int>(vecFromMat.size() * quantile)];
}

cv::Mat image_contrast(cv::Mat img) {
  double med = find_median(img);
  double low_val = med * 1.5;
  cv::Mat mask_low = (img >= low_val) / 255;
  double hig_val = find_quantile(img, 0.99);
  std::cout << low_val << ' ' << hig_val << std::endl;
  cv::Mat mask_hig = (img >= hig_val) / 255;
  cv::Mat masked_type =
      cv::Mat::zeros(mask_low.size[0], mask_low.size[1], CV_64FC1);
  mask_low.convertTo(masked_type, CV_64FC1);
  cv::Mat data = img.mul(masked_type) + low_val * (1 - masked_type);
  return cv::min(cv::max((data - low_val) / (hig_val - low_val) * 255, 0), 255);
}

cv::Mat gaus_filter(cv::Mat img, int kernel_size) {
  cv::Mat im_blur;
  cv::GaussianBlur(img, im_blur, cv::Size(kernel_size, kernel_size), 0);
  return im_blur;
}

cv::Mat image_to_int(cv::Mat img) {
  double low_val, hig_val;
  cv::minMaxLoc(img, &low_val, &hig_val, NULL, NULL);
  cv::Mat db_im =
      cv::min(cv::max((img - low_val) / (hig_val - low_val) * 255, 0), 255);
  cv::Mat int_im = cv::Mat::zeros(db_im.size[0], db_im.size[1], CV_8UC1);
  db_im.convertTo(int_im, CV_8UC1);
  return int_im;
}

cv::Mat read_tif(std::string path, int fl) {
  cv::Mat img = cv::imread(path, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
  if (fl == 32) {
    cv::Mat img1 = cv::Mat::zeros(img.size[0], img.size[1], CV_32FC1);
    img.convertTo(img1, CV_32FC1);
    return img1;
  }
  cv::Mat img1 = cv::Mat::zeros(img.size[0], img.size[1], CV_64FC1);
  img.convertTo(img1, CV_64FC1);
  return img1;
}

void dfs_comp1(const cv::Mat &img, std::vector<std::vector<int>> &used,
               cv::Point pt, int cur, std::vector<cv::Point> &all_pt) {
  used[pt.x][pt.y] = cur;
  all_pt.push_back(pt);
  if (int(img.at<uint8_t>(pt.x + 1, pt.y)) > 0 && !used[pt.x + 1][pt.y]) {
    cv::Point nxt;
    nxt.x = pt.x + 1;
    nxt.y = pt.y;
    dfs_comp1(img, used, nxt, cur, all_pt);
  }
  if (int(img.at<uint8_t>(pt.x, pt.y + 1)) > 0 && !used[pt.x][pt.y + 1]) {
    cv::Point nxt;
    nxt.x = pt.x;
    nxt.y = pt.y + 1;
    dfs_comp1(img, used, nxt, cur, all_pt);
  }
  if (int(img.at<uint8_t>(pt.x - 1, pt.y)) > 0 && !used[pt.x - 1][pt.y]) {
    cv::Point nxt;
    nxt.x = pt.x - 1;
    nxt.y = pt.y;
    dfs_comp1(img, used, nxt, cur, all_pt);
  }
  if (int(img.at<uint8_t>(pt.x, pt.y - 1)) > 0 && !used[pt.x][pt.y - 1]) {
    cv::Point nxt;
    nxt.x = pt.x;
    nxt.y = pt.y - 1;
    dfs_comp1(img, used, nxt, cur, all_pt);
  }
  if (int(img.at<uint8_t>(pt.x + 1, pt.y + 1)) > 0 &&
      !used[pt.x + 1][pt.y + 1]) {
    cv::Point nxt;
    nxt.x = pt.x + 1;
    nxt.y = pt.y + 1;
    dfs_comp1(img, used, nxt, cur, all_pt);
  }
  if (int(img.at<uint8_t>(pt.x - 1, pt.y + 1)) > 0 &&
      !used[pt.x - 1][pt.y + 1]) {
    cv::Point nxt;
    nxt.x = pt.x - 1;
    nxt.y = pt.y + 1;
    dfs_comp1(img, used, nxt, cur, all_pt);
  }
  if (int(img.at<uint8_t>(pt.x - 1, pt.y - 1)) > 0 &&
      !used[pt.x - 1][pt.y - 1]) {
    cv::Point nxt;
    nxt.x = pt.x - 1;
    nxt.y = pt.y - 1;
    dfs_comp1(img, used, nxt, cur, all_pt);
  }
  if (int(img.at<uint8_t>(pt.x + 1, pt.y - 1)) > 0 &&
      !used[pt.x + 1][pt.y - 1]) {
    cv::Point nxt;
    nxt.x = pt.x + 1;
    nxt.y = pt.y - 1;
    dfs_comp1(img, used, nxt, cur, all_pt);
  }
}

std::vector<cv::Point>
get_neighbours(cv::Point pt, cv::Mat img,
               const std::vector<std::vector<int>> &used) {
  std::vector<cv::Point> neighb;
  if (img.at<uint8_t>(pt.x - 1, pt.y) > 0 && !used[pt.x - 1][pt.y]) {
    cv::Point new_pt;
    new_pt.x = pt.x - 1;
    new_pt.y = pt.y;
    neighb.push_back(new_pt);
  }
  if (img.at<uint8_t>(pt.x + 1, pt.y) > 0 && !used[pt.x + 1][pt.y]) {
    cv::Point new_pt;
    new_pt.x = pt.x + 1;
    new_pt.y = pt.y;
    neighb.push_back(new_pt);
  }
  if (img.at<uint8_t>(pt.x - 1, pt.y - 1) > 0 && !used[pt.x - 1][pt.y - 1]) {
    cv::Point new_pt;
    new_pt.x = pt.x - 1;
    new_pt.y = pt.y - 1;
    neighb.push_back(new_pt);
  }
  if (img.at<uint8_t>(pt.x - 1, pt.y + 1) > 0 && !used[pt.x - 1][pt.y + 1]) {
    cv::Point new_pt;
    new_pt.x = pt.x - 1;
    new_pt.y = pt.y + 1;
    neighb.push_back(new_pt);
  }
  if (img.at<uint8_t>(pt.x + 1, pt.y - 1) > 0 && !used[pt.x + 1][pt.y - 1]) {
    cv::Point new_pt;
    new_pt.x = pt.x + 1;
    new_pt.y = pt.y - 1;
    neighb.push_back(new_pt);
  }
  if (img.at<uint8_t>(pt.x + 1, pt.y + 1) > 0 && !used[pt.x + 1][pt.y + 1]) {
    cv::Point new_pt;
    new_pt.x = pt.x + 1;
    new_pt.y = pt.y + 1;
    neighb.push_back(new_pt);
  }
  if (img.at<uint8_t>(pt.x, pt.y - 1) > 0 && !used[pt.x][pt.y - 1]) {
    cv::Point new_pt;
    new_pt.x = pt.x;
    new_pt.y = pt.y - 1;
    neighb.push_back(new_pt);
  }
  if (img.at<uint8_t>(pt.x, pt.y + 1) > 0 && !used[pt.x][pt.y + 1]) {
    cv::Point new_pt;
    new_pt.x = pt.x;
    new_pt.y = pt.y + 1;
    neighb.push_back(new_pt);
  }
  return neighb;
}

std::vector<cv::Point> get_neighb(cv::Point pt,
                                  const std::vector<std::vector<int>> &used) {
  std::vector<cv::Point> neighb;
  if (used[pt.x - 1][pt.y]) {
    cv::Point new_pt;
    new_pt.x = pt.x - 1;
    new_pt.y = pt.y;
    neighb.push_back(new_pt);
  }
  if (used[pt.x + 1][pt.y]) {
    cv::Point new_pt;
    new_pt.x = pt.x + 1;
    new_pt.y = pt.y;
    neighb.push_back(new_pt);
  }
  if (used[pt.x - 1][pt.y - 1]) {
    cv::Point new_pt;
    new_pt.x = pt.x - 1;
    new_pt.y = pt.y - 1;
    neighb.push_back(new_pt);
  }
  if (used[pt.x - 1][pt.y + 1]) {
    cv::Point new_pt;
    new_pt.x = pt.x - 1;
    new_pt.y = pt.y + 1;
    neighb.push_back(new_pt);
  }
  if (used[pt.x + 1][pt.y - 1]) {
    cv::Point new_pt;
    new_pt.x = pt.x + 1;
    new_pt.y = pt.y - 1;
    neighb.push_back(new_pt);
  }
  if (used[pt.x + 1][pt.y + 1]) {
    cv::Point new_pt;
    new_pt.x = pt.x + 1;
    new_pt.y = pt.y + 1;
    neighb.push_back(new_pt);
  }
  if (used[pt.x][pt.y - 1]) {
    cv::Point new_pt;
    new_pt.x = pt.x;
    new_pt.y = pt.y - 1;
    neighb.push_back(new_pt);
  }
  if (used[pt.x][pt.y + 1]) {
    cv::Point new_pt;
    new_pt.x = pt.x;
    new_pt.y = pt.y + 1;
    neighb.push_back(new_pt);
  }
  return neighb;
}

FindCompResult find_comp(cv::Mat img) {
  std::vector<std::vector<int>> used(img.size[0],
                                     std::vector<int>(img.size[1]));
  std::vector<std::vector<cv::Point>> comps;
  comps.push_back({});
  int cnt = 0;
  for (int i = 0; i < img.size[0]; ++i) {
    for (int j = 0; j < img.size[1]; ++j) {
      if (img.at<uint8_t>(i, j) > 0 && !used[i][j]) {
        ++cnt;
        used[i][j] = cnt;
        std::vector<cv::Point> st;
        cv::Point fst;
        fst.x = i;
        fst.y = j;
        st.push_back(fst);
        comps.push_back({fst});
        while (!st.empty()) {
          auto el = st.back();
          st.pop_back();
          auto neighb = get_neighbours(el, img, used);
          for (auto el1 : neighb) {
            used[el1.x][el1.y] = cnt;
            st.push_back(el1);
            comps.back().push_back(el1);
          }
        }
      }
    }
  }
  return {cnt, used, comps};
}

std::vector<cv::Point>
find_longest_path(const std::vector<std::vector<int>> &used, int idx,
                  cv::Point pt) {
  for (int i = 0; i < 10; ++i) {
    cv::Point lst = pt;
    std::queue<cv::Point> q;
    q.push(pt);
    std::set<cv::Point, CompClass> used1;
    std::map<cv::Point, cv::Point, CompClass> prev;
    used1.insert(pt);
    prev[pt] = pt;
    while (!q.empty()) {
      pt = q.front();
      q.pop();
      std::vector<cv::Point> neighb;
      if (used[pt.x - 1][pt.y] > 0 && !used1.count(cv::Point(pt.x - 1, pt.y))) {
        cv::Point new_pt;
        new_pt.x = pt.x - 1;
        new_pt.y = pt.y;
        neighb.push_back(new_pt);
      }
      if (used[pt.x + 1][pt.y] > 0 && !used1.count(cv::Point(pt.x + 1, pt.y))) {
        cv::Point new_pt;
        new_pt.x = pt.x + 1;
        new_pt.y = pt.y;
        neighb.push_back(new_pt);
      }
      if (used[pt.x - 1][pt.y - 1] > 0 &&
          !used1.count(cv::Point(pt.x - 1, pt.y - 1))) {
        cv::Point new_pt;
        new_pt.x = pt.x - 1;
        new_pt.y = pt.y - 1;
        neighb.push_back(new_pt);
      }
      if (used[pt.x - 1][pt.y + 1] > 0 &&
          !used1.count(cv::Point(pt.x - 1, pt.y + 1))) {
        cv::Point new_pt;
        new_pt.x = pt.x - 1;
        new_pt.y = pt.y + 1;
        neighb.push_back(new_pt);
      }
      if (used[pt.x + 1][pt.y - 1] > 0 &&
          !used1.count(cv::Point(pt.x + 1, pt.y - 1))) {
        cv::Point new_pt;
        new_pt.x = pt.x + 1;
        new_pt.y = pt.y - 1;
        neighb.push_back(new_pt);
      }
      if (used[pt.x + 1][pt.y + 1] > 0 &&
          !used1.count(cv::Point(pt.x + 1, pt.y + 1))) {
        cv::Point new_pt;
        new_pt.x = pt.x + 1;
        new_pt.y = pt.y + 1;
        neighb.push_back(new_pt);
      }
      if (used[pt.x][pt.y - 1] > 0 && !used1.count(cv::Point(pt.x, pt.y - 1))) {
        cv::Point new_pt;
        new_pt.x = pt.x;
        new_pt.y = pt.y - 1;
        neighb.push_back(new_pt);
      }
      if (used[pt.x][pt.y + 1] > 0 && !used1.count(cv::Point(pt.x, pt.y + 1))) {
        cv::Point new_pt;
        new_pt.x = pt.x;
        new_pt.y = pt.y + 1;
        neighb.push_back(new_pt);
      }
      for (auto el : neighb) {
        q.push(el);
        used1.insert(el);
        lst = el;
        prev[el] = pt;
      }
    }
    if (i != 9) {
      pt = lst;
    } else {
      std::vector<cv::Point> path{{lst}};
      while (prev[lst] != lst) {
        lst = prev[lst];
        path.push_back(lst);
      }
      return path;
    }
  }
  return {};
}

void dfs(cv::Point pt, std::vector<std::vector<int>> &used,
         std::vector<std::vector<int>> &nums,
         std::vector<std::vector<int>> &used_pok,
         std::vector<std::vector<cv::Point>> &cycle,
         std::vector<cv::Point> &st) {
  used_pok[pt.x][pt.y] = 1;
  int i = pt.x;
  int j = pt.y;
  st.push_back(pt);
  nums[pt.x][pt.y] = st.size() - 1;
  if (used[i - 1][j] && used_pok[i - 1][j] == 1 &&
      nums[i - 1][j] + 10 < st.size()) {
    cycle.push_back(
        std::vector<cv::Point>(st.begin() + nums[i - 1][j], st.end()));
  }
  if (used[i - 1][j] && !used_pok[i - 1][j]) {
    cv::Point pt;
    pt.x = i - 1;
    pt.y = j;
    dfs(pt, used, nums, used_pok, cycle, st);
  }
  if (used[i + 1][j] && used_pok[i + 1][j] == 1 &&
      nums[i + 1][j] + 10 < st.size()) {
    cycle.push_back(
        std::vector<cv::Point>(st.begin() + nums[i + 1][j], st.end()));
  }
  if (used[i + 1][j] && !used_pok[i + 1][j]) {
    cv::Point pt;
    pt.x = i + 1;
    pt.y = j;
    dfs(pt, used, nums, used_pok, cycle, st);
  }
  if (used[i][j - 1] && used_pok[i][j - 1] == 1 &&
      nums[i][j - 1] + 10 < st.size()) {
    cycle.push_back(
        std::vector<cv::Point>(st.begin() + nums[i][j - 1], st.end()));
  }
  if (used[i][j - 1] && !used_pok[i][j - 1]) {
    cv::Point pt;
    pt.x = i;
    pt.y = j - 1;
    dfs(pt, used, nums, used_pok, cycle, st);
  }
  if (used[i][j + 1] && used_pok[i][j + 1] == 1 &&
      nums[i][j + 1] + 10 < st.size()) {
    cycle.push_back(
        std::vector<cv::Point>(st.begin() + nums[i][j + 1], st.end()));
  }
  if (used[i][j + 1] && !used_pok[i][j + 1]) {
    cv::Point pt;
    pt.x = i;
    pt.y = j + 1;
    dfs(pt, used, nums, used_pok, cycle, st);
  }
  if (used[i - 1][j - 1] && used_pok[i - 1][j - 1] == 1 &&
      nums[i - 1][j - 1] + 10 < st.size()) {
    cycle.push_back(
        std::vector<cv::Point>(st.begin() + nums[i - 1][j - 1], st.end()));
  }
  if (used[i - 1][j - 1] && !used_pok[i - 1][j - 1]) {
    cv::Point pt;
    pt.x = i - 1;
    pt.y = j;
    dfs(pt, used, nums, used_pok, cycle, st);
  }
  if (used[i - 1][j + 1] && used_pok[i - 1][j + 1] == 1 &&
      nums[i - 1][j + 1] + 10 < st.size()) {
    cycle.push_back(
        std::vector<cv::Point>(st.begin() + nums[i - 1][j + 1], st.end()));
  }
  if (used[i - 1][j + 1] && !used_pok[i - 1][j + 1]) {
    cv::Point pt;
    pt.x = i - 1;
    pt.y = j + 1;
    dfs(pt, used, nums, used_pok, cycle, st);
  }
  if (used[i + 1][j - 1] && used_pok[i + 1][j - 1] == 1 &&
      nums[i + 1][j - 1] + 10 < st.size()) {
    cycle.push_back(
        std::vector<cv::Point>(st.begin() + nums[i + 1][j - 1], st.end()));
  }
  if (used[i + 1][j - 1] && !used_pok[i + 1][j - 1]) {
    cv::Point pt;
    pt.x = i + 1;
    pt.y = j - 1;
    dfs(pt, used, nums, used_pok, cycle, st);
  }
  if (used[i + 1][j + 1] && used_pok[i + 1][j + 1] == 1 &&
      nums[i + 1][j + 1] + 10 < st.size()) {
    cycle.push_back(
        std::vector<cv::Point>(st.begin() + nums[i + 1][j + 1], st.end()));
  }
  if (used[i + 1][j + 1] && !used_pok[i + 1][j + 1]) {
    cv::Point pt;
    pt.x = i + 1;
    pt.y = j + 1;
    dfs(pt, used, nums, used_pok, cycle, st);
  }
  used_pok[i][j] = 2;
  st.pop_back();
}

std::vector<std::vector<cv::Point>>
find_cycle(std::vector<std::vector<int>> used, int idx, cv::Point pt) {
  std::vector<std::vector<cv::Point>> cycle;
  std::vector<std::vector<int>> nums(used.size(),
                                     std::vector<int>(used[0].size()));
  std::vector<cv::Point> st;
  std::vector<std::vector<int>> used_pok(used.size(),
                                         std::vector<int>(used[0].size()));
  dfs(pt, used, nums, used_pok, cycle, st);
  return cycle;
}

cv::Point2d get_rotation_angle(cv::Point2d st, cv::Point2d lst,
                               cv::Point2d fin) {
  cv::Point2d dif1 = lst - st;
  cv::Point2d dif2 = fin - lst;
  double sin_angle =
      (dif1.x * dif2.y - dif1.y * dif2.x) / cv::norm(dif1) / cv::norm(dif2);
  double cos_angle =
      (dif1.x * dif2.x + dif1.y * dif2.y) / cv::norm(dif1) / cv::norm(dif2);
  cv::Point2d pt;
  pt.x = cos_angle;
  pt.y = sin_angle;
  return pt;
}

bool can_erase_pix(cv::Mat &skelet, cv::Point pt,
                   std::vector<cv::Point> neighb) {
  skelet.at<uint8_t>(pt.x, pt.y) = 0;
  bool del = true;
  for (auto st : neighb) {
    skelet.at<uint8_t>(pt.x, pt.y) = 0;
  }
  for (auto st : neighb) {
    int cnt = 0;
    std::set<cv::Point, CompClass> st_used, st_can;
    st_can.insert(pt);
    bool res = true;
    for (int i = 0; i < 25; ++i) {
      if (st_can.empty()) {
        res = false;
        break;
      }
      auto el = *st_can.begin();
      st_can.erase(el);
      st_used.insert(el);
      for (int j = -1; j <= 1; ++j) {
        for (int k = -1; k <= 1; ++k) {
          cv::Point nxt;
          nxt.x = el.x + k;
          nxt.y = el.y + j;
          if (j == 0 && k == 0) {
            continue;
          }
          if (skelet.at<uint8_t>(nxt.x, nxt.y) > 0 && !st_can.count(nxt) &&
              !st_used.count(nxt)) {
            st_can.insert(nxt);
          }
        }
      }
    }
    if (!res) {
      del = false;
      break;
    }
  }
  skelet.at<uint8_t>(pt.x, pt.y) = 255;
  for (auto st : neighb) {
    skelet.at<uint8_t>(pt.x, pt.y) = 255;
  }
  return del;
}

double calc_line_dist(cv::Point pt1, cv::Point pt2, const cv::Mat &raw) {
  cv::LineIterator it(pt1, pt2);
  double dst = 0;
  for (int cnt = 0; cnt != it.count; ++it, ++cnt) {
    dst += 255 - int(raw.at<uint8_t>(it.pos().x, it.pos().y));
  }
  return dst;
}

void swap_procedure(std::vector<std::vector<cv::Point>> &comp,
                    std::vector<std::vector<int>> &used, int cnt) {
  for (int i = 0; i < used.size(); ++i) {
    for (int j = 0; j < used[0].size(); ++j) {
      if (used[i][j] > 0) {
        used[i][j] = cnt - used[i][j] + 1;
      }
    }
  }
  std::reverse(comp.begin() + 1, comp.end());
  for (int i = 1; i < comp.size(); ++i) {
    std::reverse(comp[i].begin(), comp[i].end());
  }
}

cv::Mat postproc_skelet(cv::Mat skelet, int threshold = 20) {
  int cnt = 0;
  std::vector<std::vector<int>> out_comp(skelet.size[0],
                                         std::vector<int>(skelet.size[1]));
  for (int i = 0; i < skelet.size[0]; ++i) {
    for (int j = 0; j < skelet.size[1]; ++j) {
      if (skelet.at<uint8_t>(i, j) > 0 && !out_comp[i][j]) {
        ++cnt;
        std::vector<cv::Point> vec_pt;
        cv::Point pt;
        pt.x = i;
        pt.y = j;
        dfs_comp1(skelet, out_comp, pt, cnt, vec_pt);
        if (vec_pt.size() <= threshold) {
          for (auto pt : vec_pt) {
            skelet.at<uint8_t>(pt.x, pt.y) = 0;
          }
        }
      }
    }
  }
  return skelet;
}

cv::Mat skeletonize(cv::Mat bin_mask) {
  cv::Mat skelet = cv::Mat::zeros(bin_mask.size[0], bin_mask.size[1], CV_8UC1);
  cv::ximgproc::thinning((bin_mask > 0) * 255, skelet,
                         cv::ximgproc::THINNING_GUOHALL);
  return skelet;
}

std::pair<int, int> calc_img_sz(std::string path) {
  cv::Mat img = read_tif(path);
  return std::make_pair(img.size[0], img.size[1]);
}

std::vector<int> create_partition(int max_num, int step) {
  std::vector<int> part;
  for (int i = step; i + step <= max_num; i += step) {
    part.push_back(i);
  }
  return part;
}

std::pair<cv::Mat, cv::Mat> downscale_recs(double downscale_factor, std::string path, std::string scroll_id, cv::Mat img1, cv::Mat img2) {
  cv::Mat img1_res, img2_res;
  cv::resize(img1, img1_res,
             cv::Size(int(round(img1.size[1] / downscale_factor)),
                      int(round(img1.size[0] / downscale_factor))));
  cv::resize(img2, img2_res,
             cv::Size(int(round(img2.size[1] / downscale_factor)),
                      int(round(img2.size[0] / downscale_factor))));
  cv::imwrite(std::filesystem::path(path) / (scroll_id + ".1.align_300dpi.png"),
              img1_res);
  cv::imwrite(std::filesystem::path(path) / (scroll_id + ".2.align_300dpi.png"),
              img2_res);
  return {img1_res, img2_res}; 
}

std::pair<int, int> read_json(std::string json_path) {
  std::ifstream in(json_path);
  std::string line;
  int fst_el, sec_el;
  while (std::getline(in, line)) {
    int pos = -1;
    for (int i = 0; i < line.size(); ++i) {
      if (line[i] == ':') {
        pos = i;
      }
    }
    if (pos == -1) {
      continue;
    }
    bool neg = false;
    if (line[pos - 2] == 'x') {
      fst_el = 0;
      for (int j = pos + 2; j < line.size(); ++j) {
        if (line[j] == ',') {
          continue;
        }
        if (line[j] == '-') {
          neg = true;
          continue;
        }
        fst_el *= 10;
        fst_el += int(line[j] - '0');
      }
      if (neg) {
        fst_el = -fst_el;
      }
    }
    if (line[pos - 2] == 'y') {
      sec_el = 0;
      for (int j = pos + 2; j < line.size(); ++j) {
        if (line[j] == ',') {
          continue;
        }
        if (line[j] == '-') {
          neg = true;
          continue;
        }
        sec_el *= 10;
        sec_el += int(line[j] - '0');
      }
      if (neg) {
        sec_el = -sec_el;
      }
    }
  }
  return {sec_el, fst_el};
}

void clear_nans(std::string path, std::string scroll_id, int st_idx,
                int fin_idx, std::vector<int> nums) {
  for (int i = st_idx; i <= fin_idx; ++i) {
    std::ifstream in(std::filesystem::path(path) / (scroll_id + "." + get_num(nums[i]) + ".line_pts.txt"));
    std::ofstream out(std::filesystem::path(path) / (scroll_id + "." + get_num(nums[i]) + ".line_pts_new.txt"));
    std::string s;
    while (std::getline(in, s)) {
      bool cont = true;
      for (int j = 2; j < s.size(); ++j) {
        if (s[j] == 'n' && s[j - 1] == 'a' && s[j - 2] == 'n') {
          cont = false;
        }
      }
      if (!cont) {
        std::cout << "NAN in file" << std::endl;
      }
      if (cont) {
        out << s << "\n";
        out.flush();
      }
    }
  }
}

uint8_t find_quantile_png(cv::Mat img, double quant) {
  std::vector<long long> vals(256);
  int cnt = 0;
  for (int i = 0; i < img.size[0]; ++i) {
    for (int j = 0; j < img.size[1]; ++j) {
      if (img.at<uint8_t>(i, j) > 0) {
          ++cnt;
          ++vals[img.at<uint8_t>(i, j)];
      }
    }
  }
  long long cnt_big = static_cast<int>(round(cnt * (1 - quant)));
  for (int i = 255; i >= 1; --i) {
    cnt_big -= vals[i];
    if (cnt_big <= 0) {
      return i;
    }
  }
  return 0;
}

cv::Mat autocontrast(cv::Mat img) {
  uint8_t val = find_quantile_png(img, 0.995);
  std::cout << int(val) << std::endl;
  img = (img >= val) * 255 + ((img < val) / 255).mul(img) / double(val) * 255;
  return img;
}

cv::Mat correct_orientation(cv::Mat img, bool hori_flip, bool vert_flip,
                            bool swap) {
  if (hori_flip) {
    cv::Mat new_img = cv::Mat::zeros(img.size[0], img.size[1], CV_8UC1);
    for (int i = 0; i < img.size[0]; ++i) {
      for (int j = 0; j < img.size[1]; ++j) {
        new_img.at<uint8_t>(i, j) = img.at<uint8_t>(i, img.size[1] - j - 1);
      }
    }
    img = new_img;
  }
  if (vert_flip) {
    cv::Mat new_img = cv::Mat::zeros(img.size[0], img.size[1], CV_8UC1);
    for (int i = 0; i < img.size[0]; ++i) {
      for (int j = 0; j < img.size[1]; ++j) {
        new_img.at<uint8_t>(i, j) = img.at<uint8_t>(img.size[0] - i - 1, j);
      }
    }
    img = new_img;
  }
  cv::Mat new_img1 = cv::Mat::zeros(img.size[1], img.size[0], CV_8UC1);
  if (swap) {
    for (int i = 0; i < img.size[0]; ++i) {
      for (int j = 0; j < img.size[1]; ++j) {
        new_img1.at<uint8_t>(j, i) = img.at<uint8_t>(i, j);
      }
    }
    return new_img1;
  }
  return img;
}

ConfigParams read_config(std::string path) {
  std::ifstream f(path);
  nlohmann::json data = nlohmann::json::parse(f);

  std::cout << "Rd cfg" << std::endl;
  ConfigParams params{data["SCROLL_ID"], data["FOLDER_PATH_SLICES"], data["SCALE_300DPI"], data["FLIP_VERT_UNFOLDING_BOOL"],
                      data["FLIP_HORI_UNFOLDING_BOOL"], data["ROTATE_90_UNFOLDING_BOOL"], data["FILE_PATH_NUM_PATH"],
                      data["FILE_PATH_SCAN_3D"], data["FOLDER_PATH_MASK"], data["FOLDER_PATH_FINAL"],
                      data["WRITE_UNNECESSARY_DETAILS"], data["FOLDER_PATH_DETAILS"]};
  std::cout << "Rd cfg-tk" << std::endl;
  return params;
}

std::vector<int> read_nums(std::string path) {
  std::ifstream in(path);
  std::vector<int> nums;
  int num;
  while (in >> num) {
    nums.push_back(num);
  }
  return nums;
}

void init_filesystem(std::string cfg_path, ConfigParams params) {
  std::filesystem::path path(cfg_path);
  std::filesystem::current_path(path.parent_path());
  if (!std::filesystem::exists(params.folder_path_final)) {
    std::filesystem::create_directory(params.folder_path_final);
  }
  if (!std::filesystem::exists(params.folder_path_details)) {
    std::filesystem::create_directory(params.folder_path_details);
  }
}

std::vector<std::string> get_dir_paths(std::string dir) {
  std::filesystem::path path(dir);
  std::vector<std::string> raw_paths;
  for (const auto & entry : std::filesystem::directory_iterator(path)) {
    std::string ext = entry.path().extension();
    if (ext == ".png" || ext == ".tif" || ext == ".tiff") {
      raw_paths.push_back(entry.path());
    }
  }
  std::sort(raw_paths.begin(), raw_paths.end());
  return raw_paths;
}
