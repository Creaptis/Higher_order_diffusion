def add(a, b):
    return a + b


def test_args(*args):
    args_list = list(args)
    print(args_list)
    print(add(*args_list))


def test_args2(**kwargs):
    args_dict = dict(**kwargs)
    print(args_dict)
    print(add(**args_dict))


if __name__ == "__main__":
    test_args(1, 2)
    test_args2(a=1, b=2)
