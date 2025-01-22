#include "st_fin_slice.h"


int automatic_find_st_fin(std::vector<std::string> mk_paths, bool st, const std::vector<int>& nums, int slice_count) {
  std::vector<int> vec_comp(nums.size());
  for (int i = 0; i < nums.size(); ++i) {
    std::string bin_mask = mk_paths[i];
    cv::Mat markup = cv::imread(bin_mask,
                                cv::IMREAD_GRAYSCALE);
    cv::Mat label_img(markup.size(), CV_32S);
    int cnt = cv::connectedComponents(markup, label_img);
    vec_comp[i] = cnt;
  }
  int st_slice = nums.back();
  int fin_slice = nums[0];
  if (st) {
    for (int i = 0; i < nums.size(); ++i) {
      bool can = true;
      for (int j = i + 1; j < nums.size(); ++j) {
        if (vec_comp[i] - vec_comp[j] >= 0.1 * (nums[j] - nums[i])) {
          can = false;
        }
      }
      if (can) {
        st_slice = i;
        std::cout << "St slice " << nums[i] << std::endl;
        break;
      }
    }
  } else {
    for (int i = nums.size() - 1; i >= 0; --i) {
      bool can = true;
      for (int j = i - 1; j >= 0; --j) {
        if (vec_comp[i] - vec_comp[j] >= 0.1 * (nums[i] - nums[j])) {
          can = false;
        }
      }
      if (can) {
        fin_slice = i;
        std::cout << "Fin slice " << nums[i] << std::endl;
        break;
      }
    }
  }
  if (st) {
    return st_slice;
  } else {
    return fin_slice;
  } 
}