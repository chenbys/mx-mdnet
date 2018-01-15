DEBUG = 0
DENET = 1
RUN = 2


def p(msg, level=0, prefix='@Chen', ):
    # level=0 for debug all
    # level=1 for debug net
    # level=2 for run
    th = 1
    if level <= th:
        print prefix + str(msg)


def com_rec(rec, kernel=3, stride=2):
    return (rec - 1) * stride + kernel


def load_pretrained():
    path = 'saved/mdnet_otb-vot15.mat'
    # norm PARAM = [N KAPPA ALPHA BETA]
