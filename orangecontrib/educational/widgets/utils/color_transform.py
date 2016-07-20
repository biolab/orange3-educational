def rgb_hash_brighter(hash, percent_brighter):
    r, g, b = hex_to_rgb(hash)
    brightness_to_add = 255 * percent_brighter // 100
    r, g, b = r + brightness_to_add, g + brightness_to_add, b + brightness_to_add
    return rgb_to_hex(tuple(min(v, 255) for v in (r, g, b)))


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
