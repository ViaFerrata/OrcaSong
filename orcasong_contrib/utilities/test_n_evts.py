#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py

for j in range(29):

    # listpath_t = '/home/saturn/capn/mppi033h/Data/input_images/ORCA_2016_115l/tight_0_100b_t_bg_classifier/data_splits/xyzt/conc_list_files/xyzt_tight_0_100b_bg_classifier_dataset_validate_'+ str(j) + '.list'
    # listpath_c = '/home/saturn/capn/mppi033h/Data/input_images/ORCA_2016_115l/tight_0_100b_t_bg_classifier/data_splits/xyzc/conc_list_files/xyzc_tight_0_100b_bg_classifier_dataset_validate_'+ str(j) + '.list'

    listpath_t = '/home/saturn/capn/mppi033h/Data/input_images/ORCA_2016_115l/tight_0_100b_t_bg_classifier/data_splits/xyzt/conc_list_files/xyzt_tight_0_100b_bg_classifier_dataset_train_'+ str(j) + '.list'
    listpath_c = '/home/saturn/capn/mppi033h/Data/input_images/ORCA_2016_115l/tight_0_100b_t_bg_classifier/data_splits/xyzc/conc_list_files/xyzc_tight_0_100b_bg_classifier_dataset_train_'+ str(j) + '.list'

    file_list_t = [line.rstrip('\n') for line in open(listpath_t)]
    file_list_c = [line.rstrip('\n') for line in open(listpath_c)]

    n_evts_t_sum = 0
    n_evts_c_sum = 0
    n_evts_c_list = []
    for i in range(len(file_list_t)):
        f_t = h5py.File(file_list_t[i])
        f_c = h5py.File(file_list_c[i])

        n_evts_t = f_t['y'].shape[0]
        n_evts_c = f_c['y'].shape[0]

        n_evts_t_sum += n_evts_t
        n_evts_c_sum += n_evts_c
        n_evts_c_list.append(n_evts_c)

        if n_evts_t != n_evts_c:
            print('Error in step ' + str(i) + ':')
            print(i)
            print(n_evts_t)
            print(n_evts_c)
            # import sys
            # sys.exit()

        f_t.close()
        f_c.close()

    print(str(j))
    print('Finish, nothing found.')

    # print('n_evts xyzt: ' + str(n_evts_t_sum))
    # print('n_evts xyzc: ' + str(n_evts_c_sum))

    # n_evts_c_conc_list = [3148, 3232, 3222, 3226, 3242, 3214, 3252, 3347, 3196, 3317, 3175, 3361, 3284, 3227, 3240, 3200, 3240, 3301, 3334, 3297, 3291, 3238, 3256, 3274, 3182, 3313, 3264, 3183, 3274, 3176, 3301, 3275, 3253, 3375, 3298, 3247, 3257, 3253, 3333, 3226, 3256, 3236, 3236, 3270, 3253, 3245, 3236, 3368, 3252, 3254, 3260, 3248, 3257, 3202, 3275, 3234, 3335, 3325, 3298, 3238, 3310, 3247, 3198, 3251, 3304, 3263, 3208, 3281, 3247, 3306, 3206, 3280, 3199, 3282, 3243, 3225, 3388, 3304, 3262, 3191, 3205, 3311, 3290, 3196, 3171, 3278, 3189, 3203, 3199, 3270, 3324, 3317, 3261, 3317, 3191, 3180, 3324, 3263, 3309, 3261, 3251, 3334, 3263, 3203, 3232, 3182, 3318, 3226, 3283, 3187, 3375, 3241, 3309, 3328, 3296, 3258, 3257, 3262, 3129, 3281, 3242, 3336, 3254, 3256, 3075, 3240, 3204, 3227, 3257, 3260, 3211, 3255, 3266, 3359, 3253, 3194, 3326, 3350, 3227, 3225, 3414, 3272, 3320, 3185, 3255, 15938, 15983, 16051, 15932, 15737, 16113, 16002, 15850, 15993, 15885, 15768, 16099, 15789, 15907, 16063, 15772, 15792, 15834, 15844, 15977, 16188, 15963, 15950, 15902, 16005, 16005, 15802, 16050, 16073, 15777, 3533, 3473, 3347, 3399, 3527, 3473, 3371, 3458, 3420, 3467, 3505, 3396, 3499, 3483, 3522, 3532, 3380, 3412, 3481, 3535, 3479, 3504, 3484, 3544, 3534, 3370, 3479, 3495, 3517, 3449, 3502, 3467, 3439, 3434, 3516, 3494, 3467, 3618, 3507, 3465, 3419, 3390, 3338, 3503, 3464, 3548, 3481, 3425, 3628, 3466, 3451, 3432, 3367, 3481, 3438, 3384, 2866, 2989, 2887, 3048, 2923, 2907, 2871, 2862, 2834, 2890, 2881, 3050, 2935, 2805, 3417, 3382, 3429, 3449, 3511, 3329, 3317, 3376, 3361, 3397, 3383, 3340, 3306, 3301, 3426, 3358, 3245, 3349, 3372, 3457, 3449, 3497, 3421, 3372, 3404, 3457, 3421, 3446, 2535, 2567, 2539, 2628, 2605, 2583, 2530, 2557, 2397, 2572, 2534, 2540, 2615, 2570, 3219, 3104, 3111, 3096, 3119, 3171, 2860, 3155, 3016, 3041, 3013, 3009, 3123, 3171, 3123, 3060, 3232, 2979, 3163, 3167, 3181, 3113, 3071, 3111, 3086, 3231, 3024, 3145, 1819, 1778, 1882, 1773, 1838, 1820, 1816, 1764, 1820, 1792, 1796, 1782, 1774, 1882]
    #
    # for i in range(len(n_evts_c_list)):
    #     if n_evts_c_list[i] != n_evts_c_conc_list[i]:
    #         print(i)
    #         print(n_evts_c_list[i])
    #         print(n_evts_c_conc_list[i])
    #         print(file_list_c[i])
