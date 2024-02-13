import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from bresenham import bresenham
import copy
from math import atan2
import matplotlib.patches as mpatches
import cv2
import sys
import os
import json


def read_rec_scan_pts(rec_pts_path, scan_pts_path):
    '''
    Читает файл с сопоставлением точек на развертке и 3d, скане и 3d
    '''
    rec_pts = []
    scan_pts = []
    with open(rec_pts_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            vals = [float(x) for x in line.split()]
            vals[0] = int(vals[0])
            vals[1] = int(vals[1])
            rec_pts.append(vals)
    with open(scan_pts_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            vals = [float(x) for x in line.split()]
            vals[0] = int(vals[0])
            vals[1] = int(vals[1])
            scan_pts.append(vals)
    return rec_pts, scan_pts
    
    
def write_rec_scan_correspondence(rec_pts, scan_pts, path):
    with open(path, "w") as f:
        for i in range(len(rec_pts)):
            f.write(str(rec_pts[i][0]) + " " + str(rec_pts[i][1]) + " " + str(scan_pts[i][0]) + " " + str(scan_pts[i][1]) + "\n")
        
        
class AffineParams:
    def __init__(self, a, b, c, d, x0, y0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x0 = x0
        self.y0 = y0
        

def affine_estimation(rec_pts, scan_pts):
    '''
    Оценивает коэффициенты гомотетии по заданному сопоставлению точек на развертке и точек на скане
    '''
    n = len(rec_pts)
    A = np.zeros((2 * n, 6))
    y = np.zeros(2 * n)
    for i in range(n):
        A[2 * i][0] = rec_pts[i][0]
        A[2 * i][1] = rec_pts[i][1]
        A[2 * i][4] = 1
        A[2 * i + 1][2] = rec_pts[i][0]
        A[2 * i + 1][3] = rec_pts[i][1]
        A[2 * i + 1][5] = 1
        y[2 * i] = scan_pts[i][0]
        y[2 * i + 1] = scan_pts[i][1]
    res, error, _, _ = np.linalg.lstsq(A, y)
    a, b, c, d, x0, y0 = res
    return AffineParams(a, b, c, d, x0, y0), error


def apply_affine(pt, aff_params):
    '''
    По точке на развертке и параметрам гомотетии возвращает образ точки на развертке
    '''
    M = np.array([[aff_params.a, aff_params.b], [aff_params.c, aff_params.d]])
    slope = np.array([aff_params.x0, aff_params.y0])
    return M @ pt + slope


def nearest_estimation(rec_pts, scan_pts, rec_pt, prev_pt=0):
    '''
    Производит оценку параметров гомотетии на основе сопоставления точек на развертке и скане
    и возвращает образ заданной точки развертки на скане.
    Вход:
    rec_pts, scan_pts - точки на развертке и скане, для которых известно сопоставление
    rec_pt - точка на развертке, для которой хочется найти образ на скане
    Выход:
    scan_pt - предсказанный образ точки rec_pt на скане
    '''
    hom_params, error = rotated_homothety_estimation(rec_pts[-prev_pt:], scan_pts[-prev_pt:])
    scan_pt = apply_homothety(rec_pt, hom_params)
    return scan_pt


def aff_nearest_estimation(rec_pts, scan_pts, rec_pt, prev_pt=0):
    '''
    Производит оценку параметров гомотетии на основе сопоставления точек на развертке и скане
    и возвращает образ заданной точки развертки на скане.
    Вход:
    rec_pts, scan_pts - точки на развертке и скане, для которых известно сопоставление
    rec_pt - точка на развертке, для которой хочется найти образ на скане
    Выход:
    scan_pt - предсказанный образ точки rec_pt на скане
    '''
    aff_params, error = affine_estimation(rec_pts[-prev_pt:], scan_pts[-prev_pt:])
    scan_pt = apply_affine(rec_pt, aff_params)
    return scan_pt


def write_rec_scan_pts_small(pts, img_small, out_path):
    '''
    Визуализирует заданный набор точек на скане или развертке
    '''
    if img_small.ndim == 2:
        img_small_new = np.zeros((img_small.shape[0], img_small.shape[1], 3))
        img_small_new[:, :, 0] = img_small
        img_small_new[:, :, 1] = img_small
        img_small_new[:, :, 2] = img_small
        img_small = img_small_new

    for pt in pts:
        img_small[pt[0], pt[1]] = [255, 0, 0]
    Image.fromarray(img_small).save(out_path)
    
    
def write_difs(difs, out_path):
    '''
    Отписывает в файл модули невязок в аффинной модели
    '''
    with open(out_path, "w") as f:
        for dif in difs:
            f.write(str(round(dif, 3)) + "\n")
    
    
def mm_measure(a):
    '''
    Переводит расстояние между 2 пикселями в разрешении 300 dpi в мм.
    '''
    return a * 0.08467#a * 0.08436


def scale_std_dif_aff(aff_params, gt, data):
    '''
    Вычисляет параметры сопоставления: коэффициент масштабирования, угло поворота,
    стандартное отклонение для длин невязок
    '''
    det = abs(aff_params.a * aff_params.d - aff_params.b * aff_params.c)
    gd = max(det, 1 / det)
    sum1 = 0
    difs = []
    for el, gt_el in zip(data, gt):
        pred = apply_affine(el, aff_params)
        dif = mm_measure(((pred[0] - gt_el[0]) ** 2 + (pred[1] - gt_el[1]) ** 2) ** 0.5)
        sum1 += dif
        difs.append(dif)
    return gd, sum1 / len(gt), difs


def write_metrics(final_path, scroll_id, gd, std, q_5, q_8):
    '''
    Отписывает и довычисляет значения метрик: модуль угла, max(scaling, 1 / scaling),
    стандартное отклонение для длин невязок,
    80% квантиль, среднее для >= 80% квантиля невязок
    '''
    print("Global distortion:", np.round(gd, 3))
    print("Mean dif:", np.round(std, 3))
    print("50%-quantile", np.round(q_5, 3))
    print("80%-quantile", np.round(q_8, 3))
    #print("80%-quantile average", np.round(q_mean, 3))
    with open(os.path.join(final_path, scroll_id + ".estimation.txt"), "w") as f:
        f.write("Global distortion: " + str(round(gd, 3)) + "\n")
        f.write("Mean dif: " + str(round(std, 3)) + " mm\n")
        f.write("50%-quantile: " + str(round(q_5, 3)) + " mm\n")
        f.write("80%-quantile: " + str(round(q_8, 3)) + " mm")
        
        
def write_aff_params(final_path, scroll_id, aff_params):
    '''
    Отписывает и довычисляет значения параметров аффинного преобразования
    '''
    with open(os.path.join(final_path, scroll_id + ".aff_params.txt"), "w") as f:
        f.write(str(round(aff_params.a, 3)) + " " + str(round(aff_params.b, 3)) + " " + str(round(aff_params.x0, 1)) + "\n")
        f.write(str(round(aff_params.c, 3)) + " " + str(round(aff_params.d, 3)) + " " + str(round(aff_params.y0, 1)) + "\n")


def visualise_distr(difs, q_5, q_8, scroll_id, final_path):
    '''
    Визуализирует распределение длин векторов невязок и квантиль(в статье 0.5 и 0.8)
    '''
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()

    counts, bins, bar = plt.hist(np.array(difs), bins=20, weights=np.zeros_like(difs) + 1. / len(difs), range=(0, 1))
    plt.clf()

    fig, ax = plt.subplots()
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0, ymax=0.44)
    plt.xlabel("$\delta$, mm")
    plt.ylabel("Frequency", rotation=0)
    ax.yaxis.set_label_coords(0.125, 0.93)
    ax.xaxis.set_label_coords(0.925, 0.08)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.margins(0.2, 0.2)

    fig.tight_layout()
    q_num = np.sum(bins <= 0.8) - 1

    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])
    arr = [counts[0]]
    for i in range(1, len(counts)):
        arr.append((counts[i] + counts[i - 1]) / 2)
    arr.append(counts[-1])
    plt.plot(bins, arr, linewidth=2)
    plt.scatter(bins, arr, zorder=10, clip_on=False, s=100)
    plt.vlines(q_8, 0, 0.40, color='blue', linestyles='dashed', linewidth=2)
    plt.vlines(q_5, 0, 0.40, color='green', linestyles='dashed', linewidth=2)
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
        plt.hlines(r, 0, 1, color='gray', alpha=0.5, linewidth=2)
    plt.savefig(os.path.join(final_path, scroll_id + ".hist.png"), bbox_inches='tight', dpi=100)
    
    
def draw_correspondence_aff(rec_pts, scan_pts,
                        rec_img_path, scan_img_path, aff_params, final_path_correspondence):
    '''
    Сопоставление между точками на скане и развертке после применения преобразования гомотетии к развертке.
    Красный цвет - больше 1 мм, синий меньше 0.5 мм, зеленый - от 0.5 мм до 1 мм.
    Вход:
    flip_vert - нужно ли вертикальное отражение для развертки
    flip_hori - нужно ли горихонтальное отражение для развертки
    rec_pts, scan_pts - точки на развертке(после отражений) и на скане
    rec_img_path, scan_img_path - изображение развертки и скана
    scale - коэффициент масштабирования
    angle - угол поворота
    slope_x, slope_y - вектор смещения
    out_path_correspondence - путь для изображения сопоставления
    '''
    rec_img = np.array(Image.open(rec_img_path)).astype(np.uint8)
    scan_img = np.array(Image.open(scan_img_path)).astype(np.uint8)
    if rec_img.ndim == 2:
        rec_img_new = np.zeros((rec_img.shape[0], rec_img.shape[1], 3)).astype(np.uint8)
        rec_img_new[:, :, 0] = rec_img
        rec_img_new[:, :, 1] = rec_img
        rec_img_new[:, :, 2] = rec_img
        rec_img = copy.deepcopy(rec_img_new)

    rec_affine = np.zeros((scan_img.shape[0], scan_img.shape[1], 3)).astype(np.uint8)
    aff_params_mat = np.array([[aff_params.d, aff_params.c, aff_params.y0], [aff_params.b, aff_params.a, aff_params.x0]])
    rec_affine = cv2.warpAffine(rec_img, aff_params_mat, (scan_img.shape[1], scan_img.shape[0]))

    dense_img = np.zeros((scan_img.shape[0],
                          scan_img.shape[1], 3)).astype(np.uint8)
    dense_img = (0.5 * rec_affine + 0.5 * scan_img).astype(np.uint8)        
    
    new_dense = np.array(Image.fromarray(dense_img).resize((dense_img.shape[1], dense_img.shape[0]))).astype(np.uint8)

    for i in range(len(scan_pts)):
        fst_pt = copy.deepcopy(scan_pts[i])
        sec_pt = apply_affine(rec_pts[i], aff_params)
        dst = mm_measure(np.sum((fst_pt - sec_pt) ** 2) ** 0.5)
        line_pts = list(bresenham(int(round(fst_pt[0])), int(round(fst_pt[1])),
                                  int(round(sec_pt[0])), int(round(sec_pt[1]))))
        if dst < 0.5:
            for pt in line_pts:
                if pt[0] >= 0 and pt[1] >= 0:
                    new_dense[min(max(pt[0], 0), new_dense.shape[0] - 1), min(max(pt[1], 0), new_dense.shape[1] - 1)] = [0, 255, 0]
        elif dst > 1:
            for pt in line_pts:
                if pt[0] >= 0 and pt[1] >= 0:
                    new_dense[min(max(pt[0], 0), new_dense.shape[0] - 1), min(max(pt[1], 0), new_dense.shape[1] - 1)] = [255, 0, 0]
        else:
            for pt in line_pts:
                if pt[0] >= 0 and pt[1] >= 0:
                    new_dense[min(max(pt[0], 0), new_dense.shape[0] - 1), min(max(pt[1], 0), new_dense.shape[1] - 1)] = [0, 0, 255]
    for pt in scan_pts:
        new_dense[min(max(pt[0], 0), new_dense.shape[0] - 1), min(max(pt[1], 0), new_dense.shape[1] - 1)] = np.array([255, 0, 255]).astype(np.uint8)
    for pt in rec_pts:
        sec_pt = apply_affine(pt, aff_params)
        new_dense[min(max(int(round(sec_pt[0])), 0), new_dense.shape[0] - 1)][min(max(int(round(sec_pt[1])), 0), new_dense.shape[1] - 1)] = np.array([255, 255, 0]).astype(np.uint8)
    Image.fromarray(new_dense).save(final_path_correspondence)
    plt.figure(figsize=(15, 30))
    color_box1 = mpatches.Patch(color='red', label='> 1mm', linewidth=12, alpha=0.5)
    color_box2 = mpatches.Patch(color='green', label='< 0.5mm', linewidth=12, alpha=0.5)
    color_box3 = mpatches.Patch(color='blue', label='<= 1mm, >= 0.5mm', linewidth=12, alpha=0.5)
    plt.legend(handles=[color_box1, color_box2, color_box3], loc='upper right')
    plt.imshow(new_dense)

    
def filter_point_correspondence(rec_3d_pts, scan_3d_pts):
    '''
    Сопоставляет точки на развертке и скане, а также фильтрует точки на скане, которые ничему не сопоставлены
    '''
    rec_pts = []
    scan_pts = []
    used = [False] * len(scan_3d_pts)
    for i in range(len(rec_3d_pts)):
        mn = 10 ** 9
        idx = -1
        for j in range(len(scan_3d_pts)):
            dst = np.sum((np.array(rec_3d_pts)[i, 2:] - np.array(scan_3d_pts)[j, 2:]) ** 2) ** 0.5
            if dst < mn:
                mn = dst
                idx = j
        if mn < 15:
            used[idx] = True
            rec_pts.append(np.array(rec_3d_pts)[i, :2].astype(int))
            scan_pts.append(np.array(scan_3d_pts)[idx, :2].astype(int))
    return rec_pts, scan_pts

    
def run(flip_vert, flip_hori, swap, scan_3d_path, scan_img_path, final_path, details_path, scroll_id, write_details):
    rec_3d_path = os.path.join(final_path, scroll_id + ".unfolding_3d.txt")
    rec_3d_pts, scan_3d_pts = read_rec_scan_pts(rec_3d_path, scan_3d_path)
    rec_pts, scan_pts = filter_point_correspondence(rec_3d_pts, scan_3d_pts)
    rec_img = np.array(Image.open(os.path.join(final_path, scroll_id + ".1.300dpi.png"))).astype(np.uint8)
    for i in range(len(rec_pts)):
        if flip_vert:
            if swap:
                rec_pts[i][0] = rec_img.shape[1] - rec_pts[i][0] - 1
            else:
                rec_pts[i][0] = rec_img.shape[0] - rec_pts[i][0] - 1
        if flip_hori:
            if swap:
                rec_pts[i][1] = rec_img.shape[0] - rec_pts[i][1] - 1
            else:
                rec_pts[i][1] = rec_img.shape[1] - rec_pts[i][1] - 1
        if swap:
            rec_pts[i][0], rec_pts[i][1] = rec_pts[i][1], rec_pts[i][0]
    if write_details:
        corr_path = os.path.join(details_path, scroll_id + ".unfolding_scan.txt")
        write_rec_scan_correspondence(rec_pts, scan_pts, corr_path)
    aff_params, error = affine_estimation(rec_pts, scan_pts)
    gd, std, dif = scale_std_dif_aff(aff_params, scan_pts, rec_pts)
    if write_details:
        write_difs(dif, os.path.join(details_path, scroll_id + ".dist.txt"))
    q_5 = np.quantile(dif, 0.5)
    q_8 = np.quantile(dif, 0.8)
    write_metrics(final_path, scroll_id, gd, std, q_5, q_8)
    write_aff_params(final_path, scroll_id, aff_params)
    visualise_distr(dif, q_5, q_8, scroll_id, final_path)
    draw_correspondence_aff(rec_pts, scan_pts,
                            os.path.join(final_path, scroll_id + ".1.300dpi.png"), 
                            scan_img_path,
                            aff_params,
                            os.path.join(final_path, scroll_id + ".1.correspondence.png"))
    draw_correspondence_aff(rec_pts, scan_pts,
                            os.path.join(final_path, scroll_id + ".2.300dpi.png"), 
                            scan_img_path,
                            aff_params,
                            os.path.join(final_path, scroll_id + ".2.correspondence.png"))
    
    
    
if __name__ == '__main__':
    config_path = sys.argv[1]
    with open(config_path) as f:
        CFG = json.load(f)
    os.chdir(os.path.dirname(config_path))
    run(CFG["FLIP_VERT_UNFOLDING_BOOL"], CFG["FLIP_HORI_UNFOLDING_BOOL"], CFG["ROTATE_90_UNFOLDING_BOOL"],
        CFG["FILE_PATH_SCAN_3D"], CFG["FILE_PATH_SCAN"], CFG["FOLDER_PATH_FINAL"], CFG["FOLDER_PATH_DETAILS"],
        CFG["SCROLL_ID"], CFG["WRITE_UNNECESSARY_DETAILS"])
    
