import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import utils

def fit_linear(x, y):
    x = x.reshape([-1, 1])
    y = y.reshape([-1, 1])
    temp = np.ones(x.shape)
    A = np.concatenate([x, temp], axis=1)
    X = np.linalg.solve(np.matmul(A.T, A), np.matmul(A.T, y))
    # x: [a, b] y = ax + b
    return X

def correct_by_pred(pred, diopter, eye_lens_dis):
    """
    pred: predicted value 
    eye_lens_dis: distance between eyeball center and lens (m)
    diopter: eyeglasses diopter (negative diopters)
    """
    tan_ret = (1 + diopter / 100 * eye_lens_dis) * np.tan(pred)
    return np.arctan(tan_ret)

def call_correct_by_pred(gt_hpose, gt_diopter, pred_gaze, eye_lens_dis=np.array([0.04, 0.04])):
    diopter = gt_diopter[0][0]
    ## consider head pose
    input_gaze = pred_gaze.copy()
    input_gaze[:, 1] -= gt_hpose[:, 1]      # convert gaze to eyeball rotation (do not use head pitch)
    cor_gaze = correct_by_pred(input_gaze, diopter, eye_lens_dis)
    cor_gaze[:, 1] += gt_hpose[:, 1]        # conver back to gaze
    return cor_gaze

def call_compute_angle_error(pred_arr, gt_arr):
    total_num = len(pred_arr)
    err_arr = []
    for i in range(total_num):
        pred = pred_arr[i]
        gt = gt_arr[i]
        angle_err = utils.compute_angle_error(pred, gt)
        err_arr.append(angle_err)
    return np.array(err_arr)

def select_sample(root, idxs=None):
    gt_gaze = np.array(root['gt'])
    gt_hpose = np.array(root['hpose'])
    gt_diopter = np.array(root['diopter'])
    pred_gaze = np.array(root['pred'])
    if idxs is None:
        return gt_gaze, gt_hpose, gt_diopter, pred_gaze
    return gt_gaze[idxs], gt_hpose[idxs], gt_diopter[idxs], pred_gaze[idxs]

def fit_and_draw(args):
    root = np.load(args.data, allow_pickle=True).item()
    idxs = list(np.arange(len(root['pred'])))
    print('Total Sample:', len(idxs))

    gt_gaze, gt_hpose, gt_diopter, pred_gaze = select_sample(root, idxs)
    print('Diopter:', np.mean(gt_diopter))

    check_dim = args.dim      # 0 for pitch, 1 for yaw

    cor_gaze = call_correct_by_pred(gt_hpose, gt_diopter, pred_gaze)

    x_arr = gt_gaze[:, check_dim] * 180 / np.pi
    y_arr = pred_gaze[:, check_dim] * 180 / np.pi
    cy_arr = cor_gaze[:, check_dim] * 180 / np.pi

    pred_gaze_err = call_compute_angle_error(pred_gaze, gt_gaze) * 180 / np.pi
    cor_gaze_err = call_compute_angle_error(cor_gaze, gt_gaze) * 180 / np.pi
    print('Naked-eye: %f' % np.mean(pred_gaze_err))
    print('Corrected: %f' % np.mean(cor_gaze_err))

    # sort
    sort_res = sorted(enumerate(x_arr), key=lambda x:x[1])
    idx = [item[0] for item in sort_res]
    x_arr = np.array(x_arr)[idx]
    y_arr = np.array(y_arr)[idx]
    cy_arr = np.array(cy_arr)[idx]

    print('Dim %d Naked-eye: %f' % (check_dim, np.mean(np.abs(y_arr - x_arr))))
    print('Dim %d Corrected: %f' % (check_dim, np.mean(np.abs(cy_arr - x_arr))))

    # fit
    param_0 = fit_linear(x_arr, y_arr)
    param_1 = fit_linear(x_arr, cy_arr)

    # draw
    if args.type == 0:     # baseline
        plt.scatter(x_arr, y_arr, c=args.color, alpha=args.alpha)
        draw_x = np.linspace(np.min(x_arr), np.max(x_arr), num=20)
        draw_y = param_0[0] * draw_x + param_0[1]
        print('Kb:', param_0[0], param_0[1])
        plt.plot(draw_x, draw_y, c=args.color, alpha=args.alpha)
    else:                  # corrected
        plt.scatter(x_arr, cy_arr, c=args.color, alpha=args.alpha)
        draw_x = np.linspace(np.min(x_arr), np.max(x_arr), num=20)
        draw_y = param_1[0] * draw_x + param_1[1]
        print('Kb:', param_1[0], param_1[1])
        plt.plot(draw_x, draw_y, c=args.color, alpha=args.alpha)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to the .npy file')
    parser.add_argument('--dim', type=int, default=1, help='0: pitch, 1: yaw')
    parser.add_argument('--type', type=int, default=0, help='0: naked-eye, 1: corrected')
    parser.add_argument('--alpha', type=float, default=1.0, help='color alpha')
    args = parser.parse_args()
    
    plt.plot([-100, 100], [-100, 100], c='black')

    if args.type == 0:
        args.color = np.array([193, 85, 85]) / 255
    else:
        args.color = np.array([70, 130, 180]) / 255     # 'steelblue'
    fit_and_draw(args)

    plt.xlabel('GT', fontsize=16)
    plt.ylabel('Pred', fontsize=16)
    if args.dim == 1:
        plt.xlim(-30, 30)
        plt.ylim(-30, 30)
    else:
        plt.xlim(-10, 40)
        plt.ylim(-10, 40)
    plt.show()
