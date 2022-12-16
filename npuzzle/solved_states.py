def zero_first(size):
    return tuple([x for x in range(size * size)])


def zero_last(size):
    lst = [x for x in range(1, size * size)]
    lst.append(0)
    return tuple(lst)



KV = {"zero_first": zero_first, "zero_last": zero_last}
