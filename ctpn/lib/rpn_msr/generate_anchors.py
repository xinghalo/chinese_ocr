import numpy as np

def generate_basic_anchors(sizes, base_size=16):
    """
    基于基础的中心点生成框体(0,0,15,15)

    :param sizes:
    :param base_size:
    :return:
    """
    # 这里是固定的(0,0,15,15)
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    # 生成框框数组(10,4)
    anchors = np.zeros((len(sizes), 4), np.int32)
    index = 0
    for h, w in sizes:
        anchors[index] = scale_anchor(base_anchor, h, w)
        index += 1
    return anchors


def scale_anchor(anchor, h, w):
    """
    取anchor的中心做为中间点，以这个中间点针对h和w为中心生成新的框框
    :param anchor:
    :param h:
    :param w:
    :return:
    """

    # 7.5 —— base anchor 宽度的中间位置
    x_ctr = (anchor[0] + anchor[2]) * 0.5
    # 7.5 —— base anchor 高度的中间位置
    y_ctr = (anchor[1] + anchor[3]) * 0.5
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin
    scaled_anchor[2] = x_ctr + w / 2  # xmax
    scaled_anchor[1] = y_ctr - h / 2  # ymin
    scaled_anchor[3] = y_ctr + h / 2  # ymax
    return scaled_anchor


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    写死了19个框体的高度，宽度是固定的16

    :param base_size:
    :param ratios:
    :param scales:
    :return:
    """
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16]
    sizes = []
    for h in heights:
        for w in widths:
            sizes.append((h, w))
    return generate_basic_anchors(sizes)

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed; embed()
