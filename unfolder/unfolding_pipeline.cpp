#include "alignment.h"
#include "correspondence.h"
#include "new_skelet_conn.h"
#include "utility.h"
#include "st_fin_slice.h"


int main(int argc, char* argv[]) {  
  
  std::string cfg_path(argv[1]);
  ConfigParams cfg = read_config(cfg_path);
  init_filesystem(cfg_path, cfg);
  
  
  auto raw_paths = get_dir_paths(cfg.folder_path_slices);
  auto mask_paths = get_dir_paths(cfg.folder_path_mask);

  for (int i = 0; i < raw_paths.size(); ++i) {
    std::cout << raw_paths[i] << std::endl;
  }

  for (int i = 0; i < mask_paths.size(); ++i) {
    std::cout << mask_paths[i] << std::endl;
  }

  auto nums = read_nums(cfg.file_path_num_path);
  for (int num: nums) {
    std::cout << num << std::endl;
  }

  std::cout << cfg.folder_path_mask << std::endl;


  int slice_count = raw_paths.size();

  std::cout << "Start slice " << nums[automatic_find_st_fin(mask_paths, true, nums, slice_count)] << std::endl;
  std::cout << "Finish slice " << nums[automatic_find_st_fin(mask_paths, false, nums, slice_count)] << std::endl;
  
  std::cout << "Slice count " << slice_count << std::endl;

  int st_idx = automatic_find_st_fin(mask_paths, true, nums, slice_count);
  int fin_idx = automatic_find_st_fin(mask_paths, false, nums, slice_count);


  std::cout << "St idx " << st_idx << std::endl;

  double downscale = 9.55;


  
  
  skelet_conn_new(raw_paths,
                    cfg.write_unnecessary_details,
                    false,
                    cfg.folder_path_details,
                    mask_paths,
                    nums,
                    st_idx,
                    fin_idx,
                    slice_count,
                    cfg.scroll_id);
   
  
  spiral_rec_nearest(st_idx, fin_idx, true,
                        cfg.scroll_id,
                        cfg.folder_path_details, nums);

  spiral_rec_nearest(st_idx, fin_idx, false,
                        cfg.scroll_id,
                        cfg.folder_path_details, nums);
  


  cv::Mat fst_img = cv::imread(std::filesystem::path(cfg.folder_path_details) / (cfg.scroll_id + ".1.align.png"), cv::IMREAD_GRAYSCALE);
  cv::Mat sec_img = cv::imread(std::filesystem::path(cfg.folder_path_details) / (cfg.scroll_id + ".2.align.png"), cv::IMREAD_GRAYSCALE);

  fst_img = autocontrast(fst_img);
  sec_img = autocontrast(sec_img);
  cv::imwrite(std::filesystem::path(cfg.folder_path_details) / (cfg.scroll_id + ".1.align_cont.png"),
              fst_img);
  cv::imwrite(std::filesystem::path(cfg.folder_path_details) / (cfg.scroll_id + ".2.align_cont.png"),
              sec_img);



  auto res = downscale_recs(downscale, cfg.folder_path_details, cfg.scroll_id, fst_img, sec_img);

  fst_img = res.first;
  sec_img = res.second;


  fst_img = correct_orientation(fst_img, cfg.flip_hori_unfolding_bool,
                                cfg.flip_vert_unfolding_bool, cfg.rotate_90_unfolding_bool);
  sec_img = correct_orientation(sec_img, cfg.flip_hori_unfolding_bool,
                                cfg.flip_vert_unfolding_bool, cfg.rotate_90_unfolding_bool);

  cv::imwrite(std::filesystem::path(cfg.folder_path_final) / (cfg.scroll_id + ".1.300dpi.png"),
              fst_img);
  
  cv::imwrite(std::filesystem::path(cfg.folder_path_final) / (cfg.scroll_id + ".2.300dpi.png"),
              sec_img);
  

  
  clear_nans(cfg.folder_path_details, cfg.scroll_id,
             st_idx, fin_idx, nums);


  build_rec_correspondence_from_scan(
      cfg.file_path_scan_3d, cfg.folder_path_final, cfg.folder_path_details, cfg.scroll_id,
      st_idx, fin_idx, downscale, nums, slice_count);
      
  
  std::cout << "Run metric_vis" << std::endl;
  std::string command = "cd unfolder/build && python ../../metrics_vis/metrics_vis.py " + cfg_path;
  std::cout << command << std::endl;
  system(command.c_str());

  return 0;
}
