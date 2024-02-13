#pragma once

#include "utility.h"


void build_rec_correspondence_from_scan(std::string scan_pt_3d_path, std::string final_path, std::string details_path,
                                        std::string scroll_id,
                                        int st_idx, int fin_idx, double downscale, std::vector<int> nums, int slice_count);

//void build_rec_correspondence_from_scan(std::string scan_pt_3d_path,
//                                        std::string line_pts_full_path,
//                                        std::string out_rec_pt_3d_path,
//                                        int st_num, int fin_num, int step,
//                                        double downscale);

void build_rec_correspondence_from_scan_fast(std::string scan_pt_3d_path,
                                             std::string line_pts_full_path,
                                             std::string out_rec_pt_3d_path,
                                             int st_num, int fin_num, int step);
