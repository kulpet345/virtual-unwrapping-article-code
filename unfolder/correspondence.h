#pragma once

#include "utility.h"


void build_rec_correspondence_from_scan(std::string scan_pt_3d_path, std::string final_path, std::string details_path,
                                        std::string scroll_id,
                                        int st_idx, int fin_idx, double downscale, std::vector<int> nums, int slice_count);
