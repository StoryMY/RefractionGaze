import os
import h5py
import cv2
import numpy as np
import argparse

def check_cpvalue(landmarks, cpvalues):
    total_num = len(landmarks)
    for i in range(total_num):
        sclera_lmk = landmarks[i][:24]  # 24
        iris_lmk = landmarks[i][24:]    # 19
        cp_iris = np.mean(iris_lmk, axis=0)
        cp_sclera = (sclera_lmk[0] + sclera_lmk[11]) / 2
        cp_diff = cp_iris - cp_sclera
        eye_width = np.max(sclera_lmk[:, 0]) - np.min(sclera_lmk[:, 0])
        cp_value = cp_diff / eye_width
        
        check_err = np.abs(cp_value - cpvalues[i])
        assert check_err[0] < 1e-5, (cp_value, cpvalues[i])
        assert check_err[1] < 1e-5, (cp_value, cpvalues[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="001", help='Path to config json')
    args = parser.parse_args()

    h5_list = os.listdir(args.dir)
    for h5_name in h5_list:
        h5_path = os.path.join(args.dir, h5_name)
        root = h5py.File(h5_path, 'r')

        images = np.asarray(root['image'])
        landmarks = np.asarray(root['landmark'])
        cpvalues = np.asarray(root['cpvalue'])
        diopters = np.asarray(root['diopter'])
        print(diopters[0])
        check_cpvalue(landmarks, cpvalues)

        H, W = 36, 60

        img_num = len(images)
        draw_img = []
        for i in range(img_num):
            img = images[i]
            assert img.shape == (H, W, 3), img.shape
            sclera_lmk = landmarks[i][:24]  # 24
            iris_lmk = landmarks[i][24:]    # 19
            for lmk in sclera_lmk:
                x = int(lmk[0] + 0.5)
                y = int(lmk[1] + 0.5)
                img[y, x] = np.array([0, 200, 0])
            for lmk in iris_lmk:
                x = int(lmk[0] + 0.5)
                y = int(lmk[1] + 0.5)
                img[y, x] = np.array([255, 255, 0])
            draw_img.append(img)
        
        # assemble
        while len(draw_img) < 60:
            blank_img = np.zeros([H, W, 3], dtype=np.uint8)
            draw_img.append(blank_img)
        row_1 = np.concatenate(draw_img[:15], axis=1)
        row_2 = np.concatenate(draw_img[15:30], axis=1)
        row_3 = np.concatenate(draw_img[30:45], axis=1)
        row_4 = np.concatenate(draw_img[45:], axis=1)
        show_img = np.concatenate([row_1, row_2, row_3, row_4], axis=0)
        show_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join('debug', h5_name[:-3] + '.png'), show_img)
        
        root.close()
