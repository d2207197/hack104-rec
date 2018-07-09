import attr


@attr.s(auto_attribs=True)
class CVFold:
    train_start: int
    test_start: int
    test_stop: int


def series_cv(length, n_splits=3, step_ratio=0.2, train_ratio=0.7):

    step_size = round(step_ratio*length /
                      (1+step_ratio*(n_splits-1)))
    chunk_size = length - step_size*(n_splits-1)
    train_size = round(chunk_size * train_ratio)
    test_size = chunk_size - train_size
    for i in range(n_splits):
        shift = i*step_size
        yield CVFold(shift, shift + train_size, shift + train_size + test_size)
