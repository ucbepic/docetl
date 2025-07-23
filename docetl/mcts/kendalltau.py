from scipy import stats

true_f1 = [
    0.424281,
    0.511,
    0.335,
    0.34,
    0.759,
    0.324,
    0.69,
    0.493,
    0.362,
    0.397,
    0.344,
    0.449,
    0.776,
]
estimated_accuracy = [
    0.5,
    0.5,
    0.35,
    0.5,
    0.575,
    0.375,
    0.535,
    0.475,
    0.525,
    0.5,
    0.515,
    0.425,
    0.635,
]
res = stats.kendalltau(true_f1, estimated_accuracy)
print(res)
