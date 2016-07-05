from colorsys import rgb_to_hsv, hsv_to_rgb

def rgb_hash_brighter(hash, percent_brighter):
    rgb = hex_to_rgb(hash)
    hsv = list(rgb_to_hsv(*rgb))
    print(hsv)
    hsv[2] = min(1, hsv[2] + percent_brighter * (1 - hsv[2])) # correct s component
    hsv[1] = max(0, hsv[1] - percent_brighter * hsv[1])
    print(hsv)
    return rgb_to_hex(tuple(map(lambda x: int(x * 255), hsv_to_rgb(*hsv))))


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(map(lambda x: x / 255, (int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
