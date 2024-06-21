

def get_kframes(ground_positions=None, pattern="square", interpolate=False):
    # ground_positions = None
    if ground_positions is not None:
        # Add frame index to ground_positions
        # k_positions = [1, 2, 3, 15, 30, 45, 60, 75, 90, 105, 120]
        # k_positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        #                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        #                 45, 60, 75, 90,
        #                 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
        # k_positions = [0, 1, 2, 3, 4, 95, 96, 97, 98, 99]
        k_positions = [ii for ii in range(1, 120, 1)]
        if 119 not in k_positions:
            k_positions.append(119)
        kframes = []
        for k_posi in k_positions:
            kframes.append((k_posi, (float(ground_positions[k_posi - 1, 0, 0]),
                                     float(ground_positions[k_posi - 1, 0,
                                                            2]))))
        return kframes

    
    if pattern == "square":
        kframes = [ ( 1,  (0.0, 0.0)),
                    (30,  (0.0, 3.0)),
                    (45,  (1.5, 3.0)),
                    (60,  (3.0, 3.0)),
                    (75,  (3.0, 1.5)),
                    (90,  (3.0, 0.0)),
                    (105, (1.5, 0.0)),
                    (119, (0.0, 0.0))
                    ]
    elif pattern == "inverse_N":
        kframes = [ ( 1,  (0.0, 0.0)),
                    # (30,  (0.0- 2.0, 3.0- 2.0)),
                    # (45,  (1.5- 2.0, 1.5- 2.0)),
                    # (60,  (3.0- 2.0, 0.0- 2.0)),
                    # (75,  (3.0- 2.0, 1.5- 2.0)),
                    # (90,  (3.0- 2.0, 3.0- 2.0)),
                    (30,  (0.0, 3.0)),
                    (45,  (1.5, 1.5)),
                    (60,  (3.0, 0.0)),
                    # (75,  (3.0, 1.5)),
                    (90,  (3.0, 3.0)),
                    # (105, (1.5, 0.0)),
                    (119, (0.0, 0.0))
                    ]
    elif pattern == "3dots":
        kframes = [ ( 1,  (0.0, 0.0)),
                    # (29,  (0.0, 2.0)),
                    # (45,  (0.0, 3.0)),
                    # (31,  (0.0, 2.0)),
                    # (59,  (2.0, 2.0)),
                    # (59,  (3.0, 3.0)),
                    (59,  (0.0, 3.0)),
                    # (89,  (3.0, 3.0)),
                    # (89,  (3.0, 0.0)),
                    # (119,  (0.0, 3.0)),
                    # (91,  (3.0, 0.0)),
                    # (105, (1.5, 0.0)),
                    (119, (3.0, 3.0))
                    ]
    elif pattern == "sdf":
        kframes = [
            (1,   (0.0, 0.0)),
            # (90,  (2.0, 3.0)),
            # (91,  (2.0, 3.0)),
            # (92,  (2.0, 3.0)),
            # (93,  (2.0, 3.0)),
            # (94,  (2.0, 3.0)),
            # (116, (3.0, 4.5)),
            # (117, (3.0, 4.5)),
            # (118, (3.0, 4.5)),
            (119, (2.0, 2.0)),
            ]
    elif pattern == "zigzag":
        kframes = [
            (1,   (0.0, 0.0)),
            (40,   (0.0,2.0)),
            (79,   (2.0, 2.0)),
            # (119,   (2.0, 3.0)),
            # (90,  (2.0, 3.0)),
            # (91,  (2.0, 3.0)),
            # (92,  (2.0, 3.0)),
            # (93,  (2.0, 3.0)),
            # (94,  (3.0, 3.0)),
            # (94,  (-1.5, 2.0)),
            (116, (2.0, 4.0)),
            # (117, (3.0, 4.5)),
            # (118, (3.0, 4.5)),
            # (119, (0.0, 0.0)),
            ]
    else:
        # kframes = [
        #     (1,   (0.0, 0.0)),
        #     (80,  (3.0, 5.0)),
        # ]
        kframes = [
            (1,   (0.0, 0.0)),
            # (30,  (0.0, 2.0)),
            # (30,  (0.0, 3.0)),
            # (45,  (1.5, 3.0)),
            # (60,  (2.2, 2.2)),
            
            (90,  (2.0, 3.0)),
            (91,  (2.0, 3.0)),
            (92,  (2.0, 3.0)),
            (93,  (2.0, 3.0)),
            (94,  (2.0, 3.0)),

            # (60,  (0.0, 3.0)),
            # (75,  (2.5, 4)),
            # (120,  (0.0, 4.0)),
            # (90,  (3.0, 4.0)),
            # (91,  (3.0, 4.0)),
            # (92,  (3.0, 4.0)),
            # (93,  (3.0, 4.0)),
            # (105, (1.5, 0.0)),
            (116, (3.0, 4.5)),
            (117, (3.0, 4.5)),
            (118, (3.0, 4.5)),
            (119, (3.0, 4.5)),
            # (180, (-3.0, 4.0)),
            # (196, (-3.0, 6.0)),
        ]
    
    # if interpolate:
    #     kframes = interpolate_kps(kframes)
    return kframes


def get_obstacles():
    # Obstacles for obstacle avoidance task. Each one is a circle with radius
    # on the xz plane with center at (x, z)
    obs_list = [
        # ((-0.2, 3.5) , 0.5),
        ((4, 1.5) , 0.7),
        ((0.7, 1.5) , 0.6),
    ]
    return obs_list


def interpolate_kps(kframes):
    kframes_new = []
    lastx, lasty = 0.0, 0.0
    last = 0
    for frame, loc in kframes:
        diff = frame - last
        for i in range(diff):
            kframes_new.append((last + i, (lastx + (loc[0] - lastx) * i / diff , lasty + (loc[1] - lasty) * i / diff)))
        # kframes_new.append((frame, loc))
        lastx, lasty = loc
        last = frame
    # Add the last frame
    kframes_new.append((frame, loc))
    kframes = kframes_new
    return kframes