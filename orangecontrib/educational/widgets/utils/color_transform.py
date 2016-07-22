from colorsys import rgb_to_hls, hls_to_rgb


def rgb_hash_brighter(hash, percent_brighter):
    rgb = hex_to_rgb(hash)
    hls = list(rgb_to_hls(*rgb))
    hls[1] = min(1, hls[1] + percent_brighter * (1 - hls[1]))
    return rgb_to_hex(tuple(map(lambda x: int(x * 255), hls_to_rgb(*hls))))


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(map(lambda x: x / 255, (int(value[i:i + lv // 3], 16)
                                         for i in range(0, lv, lv // 3))))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
